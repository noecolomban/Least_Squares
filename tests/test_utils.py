import pytest
import os
import numpy as np
from src.utils import (
    clean_keys_json,
    restore_integer_keys,
    save_optimization_results,
    read_optimization_results,
    save_risk_results,
    read_risk_results
)

def test_clean_keys_json():
    """Test that numeric keys are properly converted to strings for JSON."""
    data = {1: "a", 2.5: "b", "c": {3: "d"}}
    cleaned = clean_keys_json(data)
    assert cleaned == {"1": "a", "2.5": "b", "c": {"3": "d"}}

def test_restore_integer_keys():
    """Test that string keys representing integers are properly restored as integers."""
    data = {"1": "a", "c": {"3": "d"}, "not_an_int": "e"}
    restored = restore_integer_keys(data)
    assert restored == {1: "a", "c": {3: "d"}, "not_an_int": "e"}

def test_optimization_results_io(monkeypatch, tmp_path):
    """Test saving and reading optimization results."""
    # Change the current directory to avoid writing to the real project folder
    monkeypatch.chdir(tmp_path)
    
    # `save_optimization_results` does not create the destination folder, we must do it
    os.makedirs(os.path.join("saved_files", "base_lr"), exist_ok=True)
    
    data = {1: {"best_eta": 0.1}, "sched": {"best_eta": 0.05}}
    filename = "test_opt.json"
    
    save_optimization_results(data, filename)
    loaded_data = read_optimization_results(filename)
    
    # Verify that the data is identical, with key restoration
    assert loaded_data == data
    assert loaded_data[1]["best_eta"] == 0.1
    assert loaded_data["sched"]["best_eta"] == 0.05

def test_risk_results_io(monkeypatch, tmp_path):
    """Test saving and reading risks (including Numpy <-> JSON conversion)."""
    monkeypatch.chdir(tmp_path)
    # Note: save_risk_results automatically creates the 'saved_files/risks' folder
    
    data = {
        "sched1": np.array([1.0, 2.0, 3.0]),
        "nested": {
            "arr1": np.array([4.0, 5.0]),
            "list_arr": [np.array([6.0]), np.array([7.0])]
        }
    }
    filename = "test_risk.json"
    
    save_risk_results(data, filename)
    loaded = read_risk_results(filename)
    
    # Verify that the type has been properly converted back to np.ndarray when reading
    assert isinstance(loaded["sched1"], np.ndarray)
    assert np.array_equal(loaded["sched1"], data["sched1"])
    assert isinstance(loaded["nested"]["arr1"], np.ndarray)
    assert np.array_equal(loaded["nested"]["list_arr"][0], data["nested"]["list_arr"][0])