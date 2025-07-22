from dataclasses import dataclass
import os


@dataclass
class LLMConfig:
    provider: str
    model: str
    temperature: float

CODEGEN_LLM_CONFIG = LLMConfig(
    provider=os.getenv("CODEGEN_LLM_PROVIDER"),
    model=os.getenv("CODEGEN_LLM"),
    temperature=float(os.getenv("CODEGEN_LLM_TEMPERATURE"))
)

TEXT_CLEAN_LLM_CONFIG = LLMConfig(
    provider=os.getenv("TEXT_CLEAN_LLM_PROVIDER"),
    model=os.getenv("TEXT_CLEAN_LLM"),
    temperature=float(os.getenv("TEXT_CLEAN_LLM_TEMPERATURE"))
)
