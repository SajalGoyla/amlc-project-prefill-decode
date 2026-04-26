#!/usr/bin/env python3
"""
GCP GPU Availability Checker
=============================
Situation: A startup without GCP premium support urgently needs a GPU VM.
           Availability must be determined dynamically across all zones.

Two methods are compared:

  Method A — Traditional
    Iterates zones one-by-one. For each zone makes individual API calls to
    check GPU support, then quota, then attempts VM creation. Sequential
    throughout. Represents the naive, unoptimized approach.

    API calls: 1 (zones list)
             + per zone: 1 (GPU check) + 1 (quota check) + 1 (VM create)

  Method B — Optimized
    Two pre-flight batch calls replace all per-zone reads. GPU support and
    quota for every zone worldwide are fetched in 2 API calls total, then
    filtered in-memory (including a hard price cap). VM creation attempts
    fire in parallel with a stop-on-first-success signal.

    API calls: 1 (global GPU discovery) + 1 (batch quota) + K parallel VM creates

Pricing filter (--price-limit, default $1.00/hr):
  T4  = $0.54/hr  → passes
  V100= $2.86/hr  → fails → logged as "Pricing too high"

Usage:
    pip install tabulate
    export GCP_PROJECT_ID="your-project"
    python gcp_gpu_checker.py --method both
    python gcp_gpu_checker.py --method both --price-limit 3.00   # allow V100
    python gcp_gpu_checker.py --method both --keep               # keep VM alive
"""

import os, sys, subprocess, json, time, argparse, logging, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
from collections import defaultdict

try:
    from tabulate import tabulate
except ImportError:
    print("ERROR: pip install tabulate")
    sys.exit(1)

# ── GPU config ────────────────────────────────────────────────────────────────
# T4 requires the N1 machine family
GPU_MACHINE: dict[str, str] = {
    "nvidia-tesla-t4": "n1-standard-4",
}
GPU_TYPES = list(GPU_MACHINE.keys())

GPU_QUOTA_KEY: dict[str, str] = {
    "nvidia-tesla-t4": "NVIDIA_T4_GPUS",
}

# ── Pricing (on-demand, approx US regions) ────────────────────────────────────
# Source: cloud.google.com/compute/vm-instance-pricing
# Failed VM creates are NOT billed. Successful VMs bill per-second, 1-min minimum.
MACHINE_PRICE_HR: dict[str, float] = {
    "n1-standard-4": 0.19,
}
GPU_PRICE_HR: dict[str, float] = {
    "nvidia-tesla-t4": 0.35,
}
DISK_PRICE_GB_HR: float = 0.04 / (24 * 30)
BOOT_DISK_GB = "20"

# ── Zone tier ordering (lower = try first) ────────────────────────────────────
REGION_TIERS: dict[str, int] = {
    "us": 0, "northamerica": 1, "southamerica": 2,
    "europe": 3, "me": 4, "africa": 5,
    "asia": 6, "australia": 7,
}

PROJECT = os.environ.get("GCP_PROJECT_ID", "")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Thread-safe API call counter ──────────────────────────────────────────────
_api_lock  = threading.Lock()
_api_count = 0
_cost_tracker = None   # set in main()


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def run_gcloud(args: list[str], timeout: int = 120) -> tuple[int, str, str]:
    global _api_count
    with _api_lock:
        _api_count += 1
    cmd = ["gcloud"] + args + ["--project", PROJECT, "--format=json", "--quiet"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out"
    except Exception as e:
        return 1, "", str(e)


def zone_to_region(zone: str) -> str:
    return "-".join(zone.split("-")[:-1])


def tier_key(zone: str) -> int:
    prefix = zone_to_region(zone).split("-")[0]
    return REGION_TIERS.get(prefix, 99)


def hourly_cost(gpu: str) -> float:
    """Total on-demand cost per hour: GPU + machine (no disk, disk is negligible)."""
    return GPU_PRICE_HR[gpu] + MACHINE_PRICE_HR[GPU_MACHINE[gpu]]


def is_price_ok(gpu: str, price_limit: float) -> bool:
    return hourly_cost(gpu) <= price_limit


def categorize(stderr: str) -> str:
    """Map a gcloud error message to a human-readable failure category."""
    clean = [l for l in stderr.splitlines()
             if not l.strip().startswith("WARNING:") and l.strip()]
    m = "\n".join(clean).lower()
    if "quota" in m:                                            return "Quota exceeded"
    if "does not have enough" in m or "exhausted" in m:         return "Insufficient capacity"
    if "not supported" in m or "not available" in m:            return "GPU not supported in zone"
    if "not found" in m and ("accelerator" in m or "gpu" in m): return "GPU type not found"
    if "billing" in m or "pric" in m:                           return "Billing/Pricing issue"
    if "permission" in m or "forbidden" in m:                   return "Permission denied"
    for line in clean:
        line = line.strip()
        if line and not line.startswith("("):
            return f"Other: {line[:120]}"
    return "Unknown error"


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA MODEL
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ZoneResult:
    zone:       str
    gpu_type:   str
    method:     str                    # "A" or "B"
    price_hr:   float = 0.0            # computed hourly cost (no API call)
    price_ok:   Optional[bool] = None  # passed price filter?
    gpu_listed: Optional[bool] = None  # GPU hardware found in zone?
    quota_ok:   Optional[bool] = None  # project quota available?
    quota_time: float = 0.0            # time spent on quota check (Method A only)
    vm_tried:   bool  = False
    vm_created: Optional[bool] = None
    vm_time:    float = 0.0
    vm_failure: str   = ""


# ═══════════════════════════════════════════════════════════════════════════════
#  COST TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

class CostTracker:
    """Tracks spend from successful VM creates and enforces a hard budget cap."""

    def __init__(self, budget: float):
        self.budget = budget
        self._spent = 0.0
        self._lock  = threading.Lock()
        self.entries: list[dict] = []

    def _estimate(self, gpu: str, duration_s: float) -> float:
        rate_hr  = hourly_cost(gpu) + DISK_PRICE_GB_HR * int(BOOT_DISK_GB)
        billed_s = max(duration_s, 60)
        return rate_hr * billed_s / 3600

    @property
    def spent(self) -> float:
        with self._lock:
            return self._spent

    def can_afford(self, gpu: str) -> bool:
        with self._lock:
            return (self._spent + self._estimate(gpu, 60)) <= self.budget

    def record(self, zone: str, gpu: str, duration_s: float) -> float:
        cost = self._estimate(gpu, duration_s)
        with self._lock:
            self._spent += cost
            self.entries.append({"zone": zone, "gpu": gpu,
                                  "duration_s": duration_s, "cost": cost})
        log.info("  [$$] +$%.4f  (total $%.4f / $%.2f budget)", cost, self._spent, self.budget)
        return cost

    def summary_rows(self) -> list[list]:
        rows = []
        with self._lock:
            for e in self.entries:
                rows.append([e["zone"], e["gpu"].replace("nvidia-tesla-", ""),
                              f"{e['duration_s']:.1f}s", f"${e['cost']:.4f}"])
            rows.append(["", "", "TOTAL", f"${self._spent:.4f}"])
        return rows


# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED — VM CREATION
# ═══════════════════════════════════════════════════════════════════════════════

def try_create_vm(zone: str, gpu: str) -> tuple[bool, float, str]:
    """
    Attempt VM creation with the specified GPU. This is the definitive test —
    success proves quota, physical capacity, and zone support are all satisfied.
    Stops the VM after creation so it can be started again at any time.
    """
    machine = GPU_MACHINE[gpu]
    name    = f"gpuchk-{zone.replace('-', '')[:10]}-{int(time.time()) % 100000}"
    t0      = time.perf_counter()

    rc, _, stderr = run_gcloud([
        "compute", "instances", "create", name,
        "--zone",               zone,
        "--machine-type",       machine,
        "--image-family",       "debian-12",
        "--image-project",      "debian-cloud",
        "--boot-disk-size",     BOOT_DISK_GB,
        "--accelerator",        f"type={gpu},count=1",
        "--maintenance-policy", "TERMINATE",
        "--no-restart-on-failure",
        "--no-address",
    ])
    dt = time.perf_counter() - t0

    if rc == 0:
        if _cost_tracker:
            _cost_tracker.record(zone, gpu, dt)
        log.info("  [VM] Created %s in %s — stopping (can be started anytime)...", name, zone)
        run_gcloud(["compute", "instances", "stop", name, "--zone", zone])
        return True, dt, ""

    return False, dt, categorize(stderr)


# ═══════════════════════════════════════════════════════════════════════════════
#  METHOD A — Traditional: sequential per-zone checks
# ═══════════════════════════════════════════════════════════════════════════════

def _get_all_zones() -> list[str]:
    """Fetch every GCP zone. This is Method A's starting point."""
    rc, stdout, stderr = run_gcloud(["compute", "zones", "list"])
    if rc != 0:
        log.error("Failed to list zones: %s", stderr[:200])
        sys.exit(1)
    try:
        entries = json.loads(stdout) if stdout.strip() else []
        zones = [e["name"] for e in entries if e.get("status") == "UP"]
        return sorted(zones, key=tier_key)   # US-first ordering
    except (json.JSONDecodeError, KeyError):
        log.error("Failed to parse zones list")
        sys.exit(1)


def _check_gpu_in_zone(zone: str, gpu: str) -> bool:
    """
    Method A per-zone GPU support check.
    One API call per zone — this is what makes Method A expensive.
    """
    rc, stdout, _ = run_gcloud([
        "compute", "accelerator-types", "list",
        "--filter", f"name={gpu}",
        "--zones", zone,
    ])
    if rc != 0:
        return False
    try:
        entries = json.loads(stdout) if stdout.strip() else []
        return any(e.get("name") == gpu for e in entries)
    except json.JSONDecodeError:
        return False


def _per_zone_quota_check(zone: str, gpu: str) -> tuple[bool, float, str]:
    """
    Method A per-zone quota check via gcloud compute regions describe.
    One API call per zone — again, expensive when done sequentially for every zone.
    """
    region = zone_to_region(zone)
    metric = GPU_QUOTA_KEY[gpu]
    t0     = time.perf_counter()
    rc, stdout, stderr = run_gcloud(["compute", "regions", "describe", region])
    dt = time.perf_counter() - t0

    if rc != 0:
        return False, dt, categorize(stderr)
    try:
        for q in json.loads(stdout).get("quotas", []):
            if q.get("metric") == metric:
                limit = float(q.get("limit", 0))
                usage = float(q.get("usage", 0))
                if limit <= 0:
                    return False, dt, "No GPU quota allocated"
                if usage >= limit:
                    return False, dt, "Quota fully consumed"
                return True, dt, f"usage={usage:.0f}/{limit:.0f}"
        return False, dt, "GPU quota metric not found in region"
    except (json.JSONDecodeError, TypeError, KeyError):
        return False, dt, "Failed to parse quota response"


def run_method_a(
    price_limit: float,
    dry_only:    bool,
) -> tuple[list[ZoneResult], float]:
    """
    Traditional approach — iterate zones one by one, making individual API calls
    at each step. No batching, no parallelism. Checks price, GPU support, and
    quota sequentially before attempting VM creation.

    This is how a developer would naively solve the problem without GCP-specific
    optimization knowledge.
    """
    log.info("\n=== Method A: Traditional (sequential per-zone checks) ===")
    log.info("  Step 1: fetch all GCP zones (1 API call)")
    t_wall = time.perf_counter()

    zones   = _get_all_zones()
    results: list[ZoneResult] = []
    won     = False

    log.info("  Step 2: iterate zones sequentially — GPU check → price → quota → VM create")

    for zone in zones:
        if won:
            break

        for gpu in GPU_TYPES:   # T4 first (cheaper), then V100
            r = ZoneResult(zone=zone, gpu_type=gpu, method="A")
            r.price_hr = hourly_cost(gpu)

            # ── GPU hardware check FIRST (1 API call per zone) ─────────────
            # Check hardware availability before price so both GPU types
            # appear in results with gpu_listed populated — proving both
            # were evaluated even when one is filtered by price.
            log.info("  [A] %-26s %-6s  checking GPU support...", zone, gpu.split("-")[-1])
            r.gpu_listed = _check_gpu_in_zone(zone, gpu)

            if not r.gpu_listed:
                r.vm_failure = "GPU not supported in zone"
                results.append(r)
                continue

            # ── Price filter (in-memory, no API call) ──────────────────────
            if not is_price_ok(gpu, price_limit):
                r.price_ok   = False
                r.vm_failure = (f"Pricing too high "
                                f"(${r.price_hr:.2f}/hr > ${price_limit:.2f} limit)")
                log.info("  [A] %-26s %-6s  PRICE FAIL  $%.2f/hr > $%.2f limit",
                         zone, gpu.split("-")[-1], r.price_hr, price_limit)
                results.append(r)
                continue

            r.price_ok = True

            # ── Quota check (1 API call per zone) ─────────────────────────
            r.quota_ok, r.quota_time, quota_msg = _per_zone_quota_check(zone, gpu)
            log.info("  [A] %-26s %-6s  quota=%s  %.2fs",
                     zone, gpu.split("-")[-1],
                     "OK" if r.quota_ok else "FAIL", r.quota_time)

            if not r.quota_ok:
                r.vm_failure = quota_msg
                results.append(r)
                continue

            # ── VM creation (1 API call per zone) ─────────────────────────
            r.vm_tried = True
            if dry_only:
                r.vm_created = False
                r.vm_failure = "Skipped (--dry-only)"
                results.append(r)
                continue

            if _cost_tracker and not _cost_tracker.can_afford(gpu):
                r.vm_created = False
                r.vm_failure = f"Budget cap reached (${_cost_tracker.spent:.4f} spent)"
                results.append(r)
                break

            r.vm_created, r.vm_time, r.vm_failure = try_create_vm(zone, gpu)
            log.info("  [A] %-26s %-6s  VM=%s  %.2fs  %s",
                     zone, gpu.split("-")[-1],
                     "OK" if r.vm_created else "FAIL",
                     r.vm_time, r.vm_failure or "success")
            results.append(r)

            if r.vm_created:
                log.info("  ✓ Method A winner: %s [%s]", zone, gpu)
                won = True
                break

    if not won and not dry_only:
        log.info("  Method A: no VM created across all zones")

    return results, time.perf_counter() - t_wall


# ═══════════════════════════════════════════════════════════════════════════════
#  METHOD B — Optimized: batch pre-flight + parallel VM creates
# ═══════════════════════════════════════════════════════════════════════════════

def _batch_gpu_discovery() -> dict[str, list[str]]:
    """
    Single global call to accelerator-types list (no zone filter).
    Returns {zone: [gpu_types_supported]} for every zone worldwide.
    Replaces ~100+ per-zone accelerator-types calls with 1 call.
    """
    rc, stdout, stderr = run_gcloud(["compute", "accelerator-types", "list"])
    if rc != 0:
        log.error("Batch GPU discovery failed: %s", stderr[:200])
        sys.exit(1)
    try:
        entries = json.loads(stdout) if stdout.strip() else []
    except json.JSONDecodeError:
        log.error("Failed to parse accelerator-types response")
        sys.exit(1)

    zone_gpus: dict[str, list[str]] = defaultdict(list)
    for e in entries:
        zone = e.get("zone", "").split("/")[-1]
        name = e.get("name", "")
        if name in GPU_TYPES and name not in zone_gpus[zone]:
            zone_gpus[zone].append(name)
    return dict(zone_gpus)


def _batch_quota_fetch() -> dict[str, dict[str, tuple[float, float]]]:
    """
    Single call to regions list to get quota for ALL regions at once.
    Returns {region: {gpu_quota_key: (usage, limit)}}.
    Replaces ~50 individual regions describe calls with 1 call.
    """
    rc, stdout, stderr = run_gcloud(["compute", "regions", "list"])
    if rc != 0:
        log.error("Batch quota fetch failed: %s", stderr[:200])
        sys.exit(1)
    try:
        regions_data = json.loads(stdout) if stdout.strip() else []
    except json.JSONDecodeError:
        log.error("Failed to parse regions list response")
        sys.exit(1)

    result: dict[str, dict[str, tuple[float, float]]] = {}
    for region_entry in regions_data:
        region_name = region_entry.get("name", "")
        quotas: dict[str, tuple[float, float]] = {}
        for q in region_entry.get("quotas", []):
            metric = q.get("metric", "")
            if metric in GPU_QUOTA_KEY.values():
                quotas[metric] = (float(q.get("usage", 0)), float(q.get("limit", 0)))
        result[region_name] = quotas
    return result


def _quota_ok_from_batch(
    zone: str,
    gpu: str,
    batch_data: dict[str, dict[str, tuple[float, float]]],
) -> tuple[bool, str]:
    """In-memory quota lookup — zero API calls."""
    region = zone_to_region(zone)
    metric = GPU_QUOTA_KEY[gpu]
    region_quotas = batch_data.get(region, {})
    if metric not in region_quotas:
        return False, "GPU quota metric not found in region"
    usage, limit = region_quotas[metric]
    if limit <= 0:
        return False, "No GPU quota allocated"
    if usage >= limit:
        return False, "Quota fully consumed"
    return True, f"usage={usage:.0f}/{limit:.0f}"


def run_method_b(
    price_limit: float,
    concurrency: int,
    dry_only:    bool,
) -> tuple[list[ZoneResult], float]:
    """
    Optimized approach — two batch API calls replace all per-zone reads.
    GPU support and quota for all zones worldwide are fetched globally,
    then filtered in-memory (price + quota). VM creation attempts fire
    in parallel with a stop-on-first-success signal.
    """
    log.info("\n=== Method B: Optimized (batch pre-flight + parallel VM creates) ===")
    t_wall = time.perf_counter()

    # ── Pre-flight batch calls ─────────────────────────────────────────────
    log.info("  Step 1: batch GPU zone discovery (1 API call globally)")
    zone_gpus  = _batch_gpu_discovery()

    log.info("  Step 2: batch quota fetch for all regions (1 API call globally)")
    batch_quota = _batch_quota_fetch()

    # ── In-memory filter (0 API calls) ────────────────────────────────────
    log.info("  Step 3: in-memory filter — price + quota (0 API calls)")
    pre_results:  list[ZoneResult] = []   # zones filtered BEFORE VM create
    candidates:   list[ZoneResult] = []   # zones that pass all filters

    zones_sorted = sorted(zone_gpus.keys(), key=tier_key)   # US-first

    for zone in zones_sorted:
        for gpu in GPU_TYPES:   # T4 first (cheaper), then V100
            if gpu not in zone_gpus[zone]:
                continue

            r = ZoneResult(zone=zone, gpu_type=gpu, method="B")
            r.price_hr   = hourly_cost(gpu)
            r.gpu_listed = True

            # Price filter
            if not is_price_ok(gpu, price_limit):
                r.price_ok   = False
                r.vm_failure = (f"Pricing too high "
                                f"(${r.price_hr:.2f}/hr > ${price_limit:.2f} limit)")
                log.info("  [B-filter] %-26s %-6s  PRICE  $%.2f/hr",
                         zone, gpu.split("-")[-1], r.price_hr)
                pre_results.append(r)
                continue

            r.price_ok = True

            # Quota filter (in-memory)
            r.quota_ok, quota_msg = _quota_ok_from_batch(zone, gpu, batch_quota)
            if not r.quota_ok:
                r.vm_failure = quota_msg
                log.info("  [B-filter] %-26s %-6s  QUOTA  %s",
                         zone, gpu.split("-")[-1], quota_msg)
                pre_results.append(r)
                continue

            candidates.append(r)

    log.info("  Filter result: %d candidates from %d GPU-supporting zones",
             len(candidates), len(zone_gpus))

    # ── Parallel VM creates ────────────────────────────────────────────────
    log.info("  Step 4: parallel VM creates (concurrency=%d, stop on first success)",
             concurrency)

    stop  = threading.Event()
    vm_results: list[ZoneResult] = []

    def create_worker(r: ZoneResult) -> ZoneResult:
        if stop.is_set():
            r.vm_failure = "Cancelled (winner already found)"
            return r
        if dry_only:
            r.vm_failure = "Skipped (--dry-only)"
            return r
        if _cost_tracker and not _cost_tracker.can_afford(r.gpu_type):
            r.vm_failure = f"Budget cap reached (${_cost_tracker.spent:.4f} spent)"
            stop.set()
            return r

        r.vm_tried   = True
        r.vm_created, r.vm_time, r.vm_failure = try_create_vm(r.zone, r.gpu_type)
        log.info("  [B-create] %-26s %-6s  %s  %.2fs  %s",
                 r.zone, r.gpu_type.split("-")[-1],
                 "OK" if r.vm_created else "FAIL",
                 r.vm_time, r.vm_failure or "success")
        if r.vm_created:
            stop.set()
            log.info("  ✓ Method B winner: %s [%s]", r.zone, r.gpu_type)
        return r

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(create_worker, r) for r in candidates]
        for f in as_completed(futures):
            vm_results.append(f.result())

    if not any(r.vm_created for r in vm_results) and not dry_only:
        log.info("  Method B: no VM created across all candidates")

    return pre_results + vm_results, time.perf_counter() - t_wall


# ═══════════════════════════════════════════════════════════════════════════════
#  REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt_bool(val: Optional[bool]) -> str:
    if val is True:  return "Yes"
    if val is False: return "No"
    return "—"


def report(
    a_results: list[ZoneResult], a_wall: float,
    b_results: list[ZoneResult], b_wall: float,
    price_limit: float,
):
    print("\n" + "=" * 150)
    print(
        f"  GCP GPU AVAILABILITY REPORT  |  {datetime.now():%Y-%m-%d %H:%M}"
        f"  |  project: {PROJECT}"
        f"  |  price limit: ${price_limit:.2f}/hr"
        f"  |  total API calls: {_api_count}"
    )
    print("=" * 150)

    # ── Main results table ─────────────────────────────────────────────────
    hdr = [
        "Zone", "GPU", "Method",
        "Price/hr", "Price OK?",
        "GPU in Zone?", "Quota OK?", "Quota\nTime(s)",
        "VM Tried?", "VM Created?", "VM\nTime(s)",
        "Failure Reason",
    ]
    rows = []
    all_results = sorted(a_results + b_results, key=lambda r: (r.zone, r.gpu_type, r.method))
    for r in all_results:
        rows.append([
            r.zone,
            r.gpu_type.replace("nvidia-tesla-", "").upper(),
            r.method,
            f"${r.price_hr:.2f}",
            _fmt_bool(r.price_ok),
            _fmt_bool(r.gpu_listed),
            _fmt_bool(r.quota_ok),
            f"{r.quota_time:.2f}" if r.quota_time else "—",
            "Yes" if r.vm_tried  else "No",
            _fmt_bool(r.vm_created),
            f"{r.vm_time:.2f}" if r.vm_time else "—",
            r.vm_failure or "—",
        ])
    print(tabulate(rows, headers=hdr, tablefmt="grid", stralign="center"))

    # ── Method comparison ──────────────────────────────────────────────────
    def avg(vals: list[float]) -> str:
        return f"{sum(vals)/len(vals):.2f}s" if vals else "n/a"

    # Method A API call breakdown
    a_gpu_checks     = len([r for r in a_results if r.gpu_listed is not None])
    a_quota_checks   = len([r for r in a_results if r.quota_time > 0])
    a_vm_creates     = len([r for r in a_results if r.vm_tried])
    a_total_api      = 1 + a_gpu_checks + a_quota_checks + a_vm_creates
    a_vm_times       = [r.vm_time for r in a_results if r.vm_time > 0]
    a_quota_times    = [r.quota_time for r in a_results if r.quota_time > 0]
    a_wins           = [r for r in a_results if r.vm_created]

    # Method B API call breakdown
    b_vm_creates     = len([r for r in b_results if r.vm_tried])
    b_total_api      = 2 + b_vm_creates    # 1 accelerator-types + 1 regions list + VM creates
    b_vm_times       = [r.vm_time for r in b_results if r.vm_time > 0]
    b_wins           = [r for r in b_results if r.vm_created]

    comp = [
        ["Description",
         "Sequential per-zone: zones list → GPU check → quota check → VM create",
         "Batch pre-flight: global GPU discovery + batch quota → parallel VM creates"],
        ["— Discovery call",       "1 × zones list",                    "1 × accelerator-types list (global)"],
        ["— GPU support checks",   f"{a_gpu_checks} × accelerator-types (per zone)", "0 (included in discovery)"],
        ["— Quota checks",         f"{a_quota_checks} × regions describe (per zone)", "1 × regions list (all regions)"],
        ["— VM create calls",      str(a_vm_creates),                   str(b_vm_creates)],
        ["Total API calls",        str(a_total_api),                    str(b_total_api)],
        ["Avg quota check time",   avg(a_quota_times),                  "n/a (in-memory)"],
        ["Avg VM create time",     avg(a_vm_times),                     avg(b_vm_times)],
        ["Wall clock time",        f"{a_wall:.2f}s",                    f"{b_wall:.2f}s"],
        ["VM succeeded?",          "Yes" if a_wins else "No",           "Yes" if b_wins else "No"],
        ["Winner zone",            a_wins[0].zone if a_wins else "None", b_wins[0].zone if b_wins else "None"],
        ["Winner GPU",             a_wins[0].gpu_type.replace("nvidia-tesla-","").upper() if a_wins else "None",
                                   b_wins[0].gpu_type.replace("nvidia-tesla-","").upper() if b_wins else "None"],
    ]
    print("\n  METHOD COMPARISON")
    print(tabulate(comp, headers=["Metric", "Method A (Traditional)", "Method B (Optimized)"],
                   tablefmt="grid"))

    # ── Failure breakdown ──────────────────────────────────────────────────
    skip_phrases = ("Skipped", "Cancelled")
    fails: dict[str, int] = {}
    for r in a_results + b_results:
        f = r.vm_failure
        if f and not any(p in f for p in skip_phrases):
            fails[f] = fails.get(f, 0) + 1

    if fails:
        print("\n  FAILURE BREAKDOWN:")
        for reason, n in sorted(fails.items(), key=lambda x: -x[1]):
            print(f"    • {reason}: {n}")

    # ── Submission table (compact, assignment format) ─────────────────────
    # Selects a representative 10-row sample covering all outcome types:
    # pricing filter, GPU not supported, quota fail, capacity fail, success.
    print("\n" + "=" * 90)
    print("  SUBMISSION TABLE  (10-zone sample — all outcome types represented)")
    print("=" * 90)

    all_res = a_results + b_results

    def _pick_rows(results: list[ZoneResult], n: int = 10) -> list[ZoneResult]:
        """Pick a representative sample: one of each failure type + the winner."""
        seen_reasons: set[str] = set()
        picked: list[ZoneResult] = []
        # Winner first
        for r in results:
            if r.vm_created and r.gpu_listed is not False:
                picked.append(r)
                break
        # One of each distinct failure type
        for r in sorted(results, key=lambda x: (x.zone, x.gpu_type)):
            if len(picked) >= n:
                break
            reason = r.vm_failure or "—"
            if reason not in seen_reasons and not r.vm_created:
                seen_reasons.add(reason)
                picked.append(r)
        # Fill remaining slots with any untried zones
        for r in sorted(results, key=lambda x: x.zone):
            if len(picked) >= n:
                break
            if r not in picked:
                picked.append(r)
        return picked[:n]

    sample = _pick_rows(all_res)
    sub_rows = []
    for r in sample:
        total_time = r.quota_time + r.vm_time
        sub_rows.append([
            r.zone,
            r.gpu_type.replace("nvidia-tesla-", "").upper(),
            "Yes" if r.gpu_listed else ("No" if r.gpu_listed is False else "—"),
            "Yes" if r.vm_created else ("No" if r.vm_created is False else "—"),
            f"{total_time:.1f}s" if total_time > 0 else (f"{r.quota_time:.1f}s" if r.quota_time else "—"),
            (r.vm_failure or "Success")[:45],
        ])
    print(tabulate(
        sub_rows,
        headers=["Zone", "GPU", "GPU\nAvailable", "VM\nAllocated", "Time", "Failure Reason"],
        tablefmt="simple",
        stralign="left",
    ))
    print()

    # ── Pricing summary ────────────────────────────────────────────────────
    print(f"\n  PRICING FILTER  (limit: ${price_limit:.2f}/hr)")
    for gpu in GPU_TYPES:
        cost = hourly_cost(gpu)
        tag  = "PASS" if cost <= price_limit else "FAIL — Pricing too high"
        print(f"    {gpu.replace('nvidia-tesla-','').upper():6s}  ${cost:.2f}/hr  →  {tag}")

    # ── Cost summary ───────────────────────────────────────────────────────
    if _cost_tracker:
        cost_rows = _cost_tracker.summary_rows()
        if len(cost_rows) > 1:
            print("\n  COST ESTIMATE  (on-demand approx; verify at cloud.google.com/compute/vm-instance-pricing)")
            print(tabulate(cost_rows,
                           headers=["Zone", "GPU", "Billed Duration", "Est. Cost (USD)"],
                           tablefmt="grid"))
        else:
            print("\n  COST ESTIMATE: $0.0000 — no VMs successfully created")
        remaining = _cost_tracker.budget - _cost_tracker.spent
        print(f"  Budget: ${_cost_tracker.budget:.2f}  |  "
              f"Spent: ${_cost_tracker.spent:.4f}  |  "
              f"Remaining: ${remaining:.4f}\n")
    else:
        print()


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="GCP GPU Availability Checker — Traditional vs Optimized",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pricing filter (--price-limit):
  T4   = $0.54/hr  →  passes default $1.00 limit
  V100 = $2.86/hr  →  fails default $1.00 limit  (logged as "Pricing too high")
  Use --price-limit 3.00 to allow V100 through the filter.

Examples:
  python gcp_gpu_checker.py --method both              # compare both methods, T4 only
  python gcp_gpu_checker.py --method both --price-limit 3.00  # allow V100
  python gcp_gpu_checker.py --method B --keep          # optimized, keep VM alive
  python gcp_gpu_checker.py --method both --dry-only   # pre-flight only, no VM creates
        """,
    )
    ap.add_argument("--method",      choices=["A", "B", "both"], default="both",
                    help="Method to run: A=Traditional, B=Optimized, both=compare (default: both)")
    ap.add_argument("--price-limit", type=float, default=1.00,
                    help="Max hourly cost per GPU VM in USD (default: $1.00 — passes T4, blocks V100)")
    ap.add_argument("--concurrency", type=int, default=5,
                    help="Parallel VM create workers for Method B (default: 5)")
    ap.add_argument("--dry-only",    action="store_true",
                    help="Run pre-flight checks only, skip all VM creation")
    ap.add_argument("--budget",      type=float, default=2.00,
                    help="Hard spend cap in USD across all VM creates (default: $2.00)")
    args = ap.parse_args()

    if not PROJECT:
        print("ERROR: Set GCP_PROJECT_ID first:  export GCP_PROJECT_ID=<your-project>")
        sys.exit(1)
    try:
        subprocess.run(["gcloud", "--version"], capture_output=True, timeout=10)
    except FileNotFoundError:
        print("ERROR: gcloud CLI not found. Install the Google Cloud SDK.")
        sys.exit(1)

    global _cost_tracker
    _cost_tracker = CostTracker(args.budget)

    log.info("=" * 70)
    log.info("GCP GPU Checker  |  project: %s", PROJECT)
    log.info("method: %s  |  price-limit: $%.2f/hr  |  concurrency: %d",
             args.method, args.price_limit, args.concurrency)
    log.info("dry-only: %s  |  budget: $%.2f",
             args.dry_only, args.budget)
    log.info("GPU pricing:  T4=$%.2f/hr", hourly_cost("nvidia-tesla-t4"))
    log.info("=" * 70)

    a_results, a_wall = [], 0.0
    b_results, b_wall = [], 0.0

    if args.method in ("A", "both"):
        a_results, a_wall = run_method_a(args.price_limit, args.dry_only)

    if args.method in ("B", "both"):
        b_results, b_wall = run_method_b(
            args.price_limit, args.concurrency, args.dry_only)

    report(a_results, a_wall, b_results, b_wall, args.price_limit)


if __name__ == "__main__":
    main()
