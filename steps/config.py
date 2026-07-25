from dataclasses import dataclass

@dataclass
class ModelNameConfig:
    model_name: str = "lightgbm"
    fine_tune: bool = False

@dataclass
class DataCleaningConfig:
    test_size: float = 0.2
    random_state: int = 42
