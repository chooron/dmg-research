"""Structure and global basin sampling modules for shared-dPL."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from dfuse import StructureSpec, enumerate_structures, get_structure


class ShuffledStructureSampler:
    """Visits each configured legal structure exactly once per cycle.

    Each cycle generates a new pseudo-random permutation from an independent
    structure RNG stream. Cycle count, position cursor, and per-structure
    exposure counts are persisted for deterministic checkpoint/resume.
    """

    def __init__(
        self,
        structures: Sequence[int | StructureSpec] | str | None = None,
        *,
        seed: int = 20260901,
    ) -> None:
        if structures is None or (isinstance(structures, str) and structures == "structures_78"):
            self.structures = tuple(int(s.model_id) for s in enumerate_structures())
        elif isinstance(structures, str):
            raise ValueError(f"unknown structure registry: {structures}")
        else:
            self.structures = tuple(int(s.model_id) if isinstance(s, StructureSpec) else int(s) for s in structures)

        if len(self.structures) < 1:
            raise ValueError("structure registry must contain at least one structure")

        self.seed = seed
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)

        self.cycle_index: int = 0
        self.cursor: int = 0
        self.exposure_counts: dict[int, int] = {struct_id: 0 for struct_id in self.structures}
        self.current_permutation: list[int] = self._new_cycle_permutation()

    def _new_cycle_permutation(self) -> list[int]:
        n = len(self.structures)
        perm = torch.randperm(n, generator=self.generator).tolist()
        return [self.structures[idx] for idx in perm]

    def next_structure(self) -> int:
        """Return the next structure ID in the shuffled cycle (K=1)."""
        if self.cursor >= len(self.current_permutation):
            self.cycle_index += 1
            self.cursor = 0
            self.current_permutation = self._new_cycle_permutation()

        struct_id = self.current_permutation[self.cursor]
        self.cursor += 1
        self.exposure_counts[struct_id] += 1
        return struct_id

    def state_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "structures": list(self.structures),
            "cycle_index": self.cycle_index,
            "cursor": self.cursor,
            "current_permutation": list(self.current_permutation),
            "exposure_counts": dict(self.exposure_counts),
            "generator_state": self.generator.get_state(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.seed = int(state["seed"])
        self.structures = tuple(int(s) for s in state["structures"])
        self.cycle_index = int(state["cycle_index"])
        self.cursor = int(state["cursor"])
        self.current_permutation = list(state["current_permutation"])
        self.exposure_counts = {int(k): int(v) for k, v in state["exposure_counts"].items()}
        gen_state = state["generator_state"]
        if isinstance(gen_state, torch.Tensor):
            gen_state = gen_state.cpu().to(torch.uint8)
        self.generator.set_state(gen_state)


class GlobalBasinSampler:
    """Single global basin permutation and cursor shared across all structure steps.

    Hard invariants:
    - Basin cursor is strictly shared across all structure steps (no per-structure cursor).
    - Basin cursor does NOT reset at structure-cycle boundaries.
    - Exhausted basin permutations trigger a reshuffle using an independent basin RNG stream.
    - Marginal basin exposures remain balanced.
    - State is fully serializable and deterministically resumable.
    """

    def __init__(
        self,
        basin_ids: Sequence[str | int],
        batch_size: int = 100,
        *,
        seed: int = 20260902,
    ) -> None:
        if len(basin_ids) < 1:
            raise ValueError("basin_ids must contain at least one basin")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        self.basin_ids = tuple(str(b) for b in basin_ids)
        self.batch_size = batch_size
        self.seed = seed

        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)

        self.basin_cycle_index: int = 0
        self.cursor: int = 0
        self.exposure_counts: dict[str, int] = {b: 0 for b in self.basin_ids}
        self.current_permutation: list[str] = self._new_cycle_permutation()

    def _new_cycle_permutation(self) -> list[str]:
        n = len(self.basin_ids)
        perm = torch.randperm(n, generator=self.generator).tolist()
        return [self.basin_ids[idx] for idx in perm]

    def next_batch(self) -> tuple[str, ...]:
        """Return the next basin batch of exact size batch_size without within-batch duplicates."""
        if self.batch_size > len(self.basin_ids):
            raise ValueError(f"batch_size ({self.batch_size}) exceeds total basins ({len(self.basin_ids)})")

        batch: list[str] = []
        batch_set: set[str] = set()

        while len(batch) < self.batch_size:
            if self.cursor >= len(self.current_permutation):
                self.basin_cycle_index += 1
                self.cursor = 0
                self.current_permutation = self._new_cycle_permutation()

            needed = self.batch_size - len(batch)

            if not batch_set:
                remaining = len(self.current_permutation) - self.cursor
                take_count = min(needed, remaining)
                slice_items = self.current_permutation[self.cursor : self.cursor + take_count]
                batch.extend(slice_items)
                batch_set.update(slice_items)
                self.cursor += take_count
            else:
                available_in_new = self.current_permutation[self.cursor :]
                candidates = [b for b in available_in_new if b not in batch_set]
                deferred = [b for b in available_in_new if b in batch_set]

                taken = candidates[:needed]
                remaining_candidates = candidates[needed:]

                batch.extend(taken)
                batch_set.update(taken)

                self.current_permutation = (
                    self.current_permutation[: self.cursor]
                    + taken
                    + deferred
                    + remaining_candidates
                )
                self.cursor += len(taken)

        for b in batch:
            self.exposure_counts[b] += 1

        return tuple(batch)

    def state_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "basin_ids": list(self.basin_ids),
            "batch_size": self.batch_size,
            "basin_cycle_index": self.basin_cycle_index,
            "cursor": self.cursor,
            "current_permutation": list(self.current_permutation),
            "exposure_counts": dict(self.exposure_counts),
            "generator_state": self.generator.get_state(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.seed = int(state["seed"])
        self.basin_ids = tuple(str(b) for b in state["basin_ids"])
        self.batch_size = int(state["batch_size"])
        self.basin_cycle_index = int(state["basin_cycle_index"])
        self.cursor = int(state["cursor"])
        self.current_permutation = list(state["current_permutation"])
        self.exposure_counts = {str(k): int(v) for k, v in state["exposure_counts"].items()}
        gen_state = state["generator_state"]
        if isinstance(gen_state, torch.Tensor):
            gen_state = gen_state.cpu().to(torch.uint8)
        self.generator.set_state(gen_state)
