"""
workloads.py — Benchmark workload profiles.

Each profile defines input/output token targets and generates prompts
by sampling from a source text to hit the desired token count.
"""

import os
import random
from dataclasses import dataclass
from pathlib import Path

# Lazy-load tokenizer to avoid import overhead when not needed
_tokenizer = None


def get_tokenizer(model_name: str = "meta-llama/Llama-3.2-3B-Instruct"):
    """Load the model tokenizer (cached after first call)."""
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=os.environ.get("HF_TOKEN"),
        )
    return _tokenizer


def count_tokens(text: str, model_name: str = "meta-llama/Llama-3.2-3B-Instruct") -> int:
    """Count tokens in text using the model tokenizer."""
    tok = get_tokenizer(model_name)
    return len(tok.encode(text))


def load_source_text() -> str:
    """Load source text for prompt generation."""
    pg_path = Path(__file__).parent / "prompts" / "pg_essays.txt"
    if pg_path.exists():
        return pg_path.read_text(encoding="utf-8")

    # Fallback: generate deterministic filler text
    paragraphs = [
        "The most important thing in any organization is the quality of the people. "
        "Technology comes and goes, but the insights of a talented team persist. "
        "When building systems at scale, one must consider not just the immediate "
        "requirements but also the evolutionary path of the architecture.",

        "Machine learning has fundamentally changed how we approach optimization "
        "problems. Where we once needed hand-crafted heuristics, we now train models "
        "that learn the structure of the problem space. This shift from explicit "
        "programming to learned representations is perhaps the most significant "
        "paradigm change in computer science since the invention of high-level "
        "programming languages.",

        "Distributed systems introduce a fundamentally different set of challenges "
        "compared to single-machine programs. Network partitions, clock skew, and "
        "partial failures are not edge cases but the normal operating conditions. "
        "Any system that does not account for these realities will eventually fail "
        "in surprising and often catastrophic ways.",

        "The economics of cloud computing have shifted the calculus of build versus "
        "buy decisively toward buy for most commodity infrastructure. However, the "
        "differentiated components of a system — the parts that give a product its "
        "unique value — still benefit from custom engineering. The art is in knowing "
        "where to draw the line.",

        "Large language models represent a curious inversion of the traditional "
        "software development process. Instead of writing explicit logic that handles "
        "specific cases, we provide examples and the model learns to generalize. "
        "This is both more powerful and more unpredictable than traditional approaches. "
        "The challenge of inference serving at scale is making this unpredictability "
        "manageable while maintaining low latency and high throughput.",
    ]
    # Repeat to get enough source text
    return "\n\n".join(paragraphs * 50)


def make_prompt(target_tokens: int, source_text: str, model_name: str) -> tuple[str, int]:
    """Create a prompt that is approximately target_tokens long.

    Returns (prompt_text, actual_token_count).
    """
    tok = get_tokenizer(model_name)
    tokens = tok.encode(source_text)

    # Random offset for variety
    max_offset = max(0, len(tokens) - target_tokens - 100)
    offset = random.randint(0, max_offset) if max_offset > 0 else 0

    # Take a slice of tokens
    selected = tokens[offset : offset + target_tokens]
    prompt_text = tok.decode(selected, skip_special_tokens=True)

    # Wrap in an instruction
    instruction = "Please read the following text carefully and provide a detailed summary:\n\n"
    full_prompt = instruction + prompt_text
    actual_tokens = len(tok.encode(full_prompt))

    return full_prompt, actual_tokens


@dataclass
class WorkloadProfile:
    """Definition of a benchmark workload."""
    name: str
    input_tokens: int
    output_tokens: int
    description: str


# ── Standard profiles from the proposal ──────────────────────────────────────
CHAT_PROFILE = WorkloadProfile(
    name="chat",
    input_tokens=50,
    output_tokens=500,
    description="Short prompt, long generation (chat-like)",
)

DOC_QA_PROFILE = WorkloadProfile(
    name="doc_qa",
    input_tokens=2000,
    output_tokens=100,
    description="Long document, short answer (doc Q&A)",
)

BALANCED_PROFILE = WorkloadProfile(
    name="balanced",
    input_tokens=500,
    output_tokens=200,
    description="Medium prompt, medium generation",
)

# ── Sweep profile for crossover analysis ────────────────────────────────────
SWEEP_INPUT_TOKENS = [50, 100, 200, 500, 1000, 1500, 2000, 3000, 4000]
SWEEP_OUTPUT_TOKENS = 100  # Fixed output length for sweep

ALL_PROFILES = [CHAT_PROFILE, DOC_QA_PROFILE, BALANCED_PROFILE]


def generate_workload(
    profile: WorkloadProfile,
    num_requests: int,
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
    seed: int = 42,
) -> list[tuple[str, int, int]]:
    """Generate a list of (prompt_text, prompt_tokens, max_output_tokens)."""
    random.seed(seed)
    source = load_source_text()
    workload = []

    for _ in range(num_requests):
        prompt, actual_tokens = make_prompt(profile.input_tokens, source, model_name)
        workload.append((prompt, actual_tokens, profile.output_tokens))

    return workload


def generate_sweep_workload(
    input_tokens: int,
    num_requests: int,
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
    seed: int = 42,
) -> list[tuple[str, int, int]]:
    """Generate sweep workload for a specific input token count."""
    profile = WorkloadProfile(
        name=f"sweep_{input_tokens}",
        input_tokens=input_tokens,
        output_tokens=SWEEP_OUTPUT_TOKENS,
        description=f"Sweep: {input_tokens} input, {SWEEP_OUTPUT_TOKENS} output",
    )
    return generate_workload(profile, num_requests, model_name, seed)
