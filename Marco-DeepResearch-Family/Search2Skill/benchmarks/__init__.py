from .base import BaseBenchmark
from .mmlu_pro import MMLUProBenchmark
from .finance_bench import FinanceBenchBenchmark
from .supergpqa import SuperGPQABenchmark
from .omnimath import OmniMathBenchmark
from .webinstruct import WebInstructBenchmark
from .rl_train import RLTrainBenchmark
from .gpqa_diamond import GPQADiamondBenchmark
from .theoremqa import TheoremQABenchmark
from .aime2025 import AIME2025Benchmark
from .aime2026 import AIME2026Benchmark
from .scibench import SciBenchBenchmark
from .chembench4k import ChemBench4KBenchmark
from .evoagentbench import EvoAgentBenchBenchmark
from .livecode_bench import LiveCodeBenchBenchmark
from .putnam_axiom import PutnamAxiomBenchmark
from .math_number_theory import MathNumberTheoryBenchmark
from .mmlu_pro_law import MMLUProLawBenchmark, MMLUProSubsetBenchmark

BENCHMARK_REGISTRY = {
    "mmlu_pro": MMLUProBenchmark,
    "finance_bench": FinanceBenchBenchmark,
    "supergpqa": SuperGPQABenchmark,
    "omnimath": OmniMathBenchmark,
    "webinstruct": WebInstructBenchmark,
    "webinstruct-math": WebInstructBenchmark,
    "rl_train": RLTrainBenchmark,
    "gpqa_diamond": GPQADiamondBenchmark,
    "theoremqa": TheoremQABenchmark,
    "aime2025": AIME2025Benchmark,
    "aime2026": AIME2026Benchmark,
    "scibench": SciBenchBenchmark,
    "chembench4k": ChemBench4KBenchmark,
    "evoagentbench": EvoAgentBenchBenchmark,
    "livecode_bench": LiveCodeBenchBenchmark,
    "putnam_axiom": PutnamAxiomBenchmark,
    "math_number_theory": MathNumberTheoryBenchmark,
    "mmlu_pro_law": MMLUProLawBenchmark,
    "mmlu_pro_engineering": MMLUProSubsetBenchmark,
    "mmlu_pro_business": MMLUProSubsetBenchmark,
    "mmlu_pro_health": MMLUProSubsetBenchmark,
    "mmlu_pro_philosophy": MMLUProSubsetBenchmark,
    "mmlu_pro_history": MMLUProSubsetBenchmark,
    "mmlu_pro_other": MMLUProSubsetBenchmark,
    "mmlu_pro_economics": MMLUProSubsetBenchmark,
    "supergpqa_physics": SuperGPQABenchmark,
    "supergpqa_chemistry": SuperGPQABenchmark,
    "supergpqa_mathematics": SuperGPQABenchmark,
    "supergpqa_hard_mathematics": SuperGPQABenchmark,
    "supergpqa_engineering": SuperGPQABenchmark,
    "supergpqa_science": SuperGPQABenchmark,
    "supergpqa_science_middle": SuperGPQABenchmark,
    "supergpqa_physics_middle": SuperGPQABenchmark,
    "supergpqa_chemistry_middle": SuperGPQABenchmark,
    "supergpqa_science_p_c": SuperGPQABenchmark,
    "science_p_c": SuperGPQABenchmark,
}


def load_benchmark(name: str) -> BaseBenchmark:
    """Factory function: return a Benchmark instance by name."""
    if name not in BENCHMARK_REGISTRY:
        raise ValueError(f"Unknown benchmark: {name}. Available: {list(BENCHMARK_REGISTRY.keys())}")
    cls = BENCHMARK_REGISTRY[name]
    # WebInstruct: webinstruct excludes math, webinstruct-math includes math
    if name == "webinstruct-math":
        return cls(exclude_math=False)
    return cls()


__all__ = ["BaseBenchmark", "load_benchmark", "BENCHMARK_REGISTRY"]
