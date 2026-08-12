"""Shared CAMELS indexing and forcing preparation for active IC runners."""

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd

from training.data_contract import FORCING_NAMES, load_dates, load_gage_ids


def gate_a_audit(config: dict) -> dict[str, tuple[int, int] | int]:
    dates = pd.to_datetime(load_dates(config["dates_path"]))
    periods = config["time_periods"]

    def bounds(name: str) -> tuple[int, int]:
        period = periods[name]
        start = pd.Timestamp(period["start"])
        end = pd.Timestamp(period["end"])
        start_index = int((dates >= start).argmax())
        end_index = len(dates) - 1 - int((dates <= end)[::-1].argmax())
        assert dates[start_index].date() == start.date()
        assert dates[end_index].date() == end.date()
        return start_index, end_index

    cal = bounds("calibration")
    evaluation = bounds("evaluation")
    assert cal[1] - cal[0] + 1 == 3652
    assert evaluation[1] - evaluation[0] + 1 == 4018
    assert evaluation[0] == cal[1] + 1

    warmup = bounds("warmup")
    assert warmup[1] == cal[0] - 1
    return {"cal": cal, "eval": evaluation, "warmup": warmup, "wd": warmup[1] - warmup[0] + 1}


def prepare_data(config: dict, time_indices: dict) -> tuple[dict, dict, np.ndarray, np.ndarray, list[str], np.ndarray]:
    data = np.load(config["data_npz"], allow_pickle=True)
    forcing = np.asarray(data["forcing"], np.float32)
    target = np.asarray(data["target"], np.float32)
    with open(config["data_basin_ids"]) as handle:
        basin_ids = [str(value).zfill(8) for value in json.load(handle)]
    with open(config["data_pkl_dataset"], "rb") as handle:
        _, _, attributes = pickle.load(handle)
    full_ids = load_gage_ids(config["gage_ids_path"])
    full_index = {basin_id: index for index, basin_id in enumerate(full_ids)}
    frac_snow = np.array([attributes[full_index[basin_id], 3] for basin_id in basin_ids])

    cal_start, cal_end = time_indices["cal"]
    warmup_days = int(time_indices["wd"])
    forcing_axis = {"precip": FORCING_NAMES.index("P"),
                    "temp": FORCING_NAMES.index("T"),
                    "pet": FORCING_NAMES.index("PET")}
    forcing_train = {
        key: forcing[cal_start - warmup_days:cal_end + 1, :, axis].transpose().copy().astype(np.float32)
        for key, axis in forcing_axis.items()
    }
    temp_train = forcing[cal_start:cal_end + 1, :, forcing_axis["temp"]]
    temp_mean_train = temp_train.mean(axis=0).astype(np.float32)
    temp_std_train = temp_train.std(axis=0).astype(np.float32)
    forcing_train["temp_mean_train"] = temp_mean_train.copy()
    forcing_train["temp_std_train"] = temp_std_train.copy()
    observations_train = target[cal_start:cal_end + 1, :, 0].transpose().copy().astype(np.float32)
    evaluation_start = cal_end - warmup_days + 1
    evaluation_end = time_indices["eval"][1]
    forcing_eval = {
        key: forcing[evaluation_start:evaluation_end + 1, :, axis].transpose().copy().astype(np.float32)
        for key, axis in forcing_axis.items()
    }
    forcing_eval["temp_mean_train"] = temp_mean_train.copy()
    forcing_eval["temp_std_train"] = temp_std_train.copy()
    observations_eval = target[time_indices["eval"][0]:evaluation_end + 1, :, 0].transpose().copy().astype(np.float32)
    return forcing_train, forcing_eval, observations_train, observations_eval, basin_ids, frac_snow
