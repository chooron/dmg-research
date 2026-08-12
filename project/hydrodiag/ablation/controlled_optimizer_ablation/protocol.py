import dataclasses
from typing import List

@dataclasses.dataclass
class Phase1Task:
    basin_id: str
    optimizer_name: str
    seed: int
    start_idx: int
    population: int
    generations: int
    stdev_init: float
    output_dir: str
    model_key: str
    center_init: list
    compute_test_metric: bool

SEEDS = [101, 202, 303]
