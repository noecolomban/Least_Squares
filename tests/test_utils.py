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
    """Teste que les clés numériques sont bien converties en chaînes de caractères pour le JSON."""
    data = {1: "a", 2.5: "b", "c": {3: "d"}}
    cleaned = clean_keys_json(data)
    assert cleaned == {"1": "a", "2.5": "b", "c": {"3": "d"}}

def test_restore_integer_keys():
    """Teste que les clés de chaînes représentant des entiers sont bien restaurées en entiers."""
    data = {"1": "a", "c": {"3": "d"}, "not_an_int": "e"}
    restored = restore_integer_keys(data)
    assert restored == {1: "a", "c": {3: "d"}, "not_an_int": "e"}

def test_optimization_results_io(monkeypatch, tmp_path):
    """Teste la sauvegarde et la lecture des résultats d'optimisation."""
    # Change le répertoire courant pour éviter d'écrire dans le vrai dossier du projet
    monkeypatch.chdir(tmp_path)
    
    # `save_optimization_results` ne crée pas le dossier de destination, on doit le faire
    os.makedirs(os.path.join("saved_files", "base_lr"), exist_ok=True)
    
    data = {1: {"best_eta": 0.1}, "sched": {"best_eta": 0.05}}
    filename = "test_opt.json"
    
    save_optimization_results(data, filename)
    loaded_data = read_optimization_results(filename)
    
    # Vérifie que les données sont identiques, avec restauration des clés
    assert loaded_data == data
    assert loaded_data[1]["best_eta"] == 0.1
    assert loaded_data["sched"]["best_eta"] == 0.05

def test_risk_results_io(monkeypatch, tmp_path):
    """Teste la sauvegarde et la lecture des risques (incluant la conversion Numpy <-> JSON)."""
    monkeypatch.chdir(tmp_path)
    # Note : save_risk_results crée automatiquement le dossier 'saved_files/risks'
    
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
    
    # Vérifie que le type a bien été reconverti en np.ndarray lors de la lecture
    assert isinstance(loaded["sched1"], np.ndarray)
    assert np.array_equal(loaded["sched1"], data["sched1"])
    assert isinstance(loaded["nested"]["arr1"], np.ndarray)
    assert np.array_equal(loaded["nested"]["list_arr"][0], data["nested"]["list_arr"][0])