import pytest
import numpy as np
from src.least_squares import LinearRegression
from src.SGD import SGD, NoisyGD
from src.risk_computations import RiskComputations
from scheduled import WSDSchedule, ConstantSchedule
from tests.test_sgd import DummySchedule  # On réutilise notre mock

@pytest.fixture
def risk_setup():
    model = LinearRegression(dim=3)
    schedules = [DummySchedule(steps=5), DummySchedule(steps=5)]
    schedules_names = ["Sched 1", "Sched 2"]
    x0 = np.random.randn(3, 1)
    return model, schedules, schedules_names, x0

@pytest.fixture
def risk_setup_with_seed():
    """Fixture avec seed pour résultats reproductibles."""
    np.random.seed(42)
    model = LinearRegression(dim=3)
    schedules = [DummySchedule(steps=10), DummySchedule(steps=10)]
    schedules_names = ["Sched 1", "Sched 2"]
    x0 = np.random.randn(3, 1)
    return model, schedules, schedules_names, x0

def test_risk_initialization(risk_setup):
    model, schedules, schedules_names, x0 = risk_setup
    risk = RiskComputations(model, x0, schedules, schedules_names, sgd_class=SGD)
    
    assert risk.model == model
    assert np.array_equal(risk.x0, x0)
    assert len(risk.schedules) == 2
    assert "Sched 1" in risk.schedules
    assert "Sched 2" in risk.schedules
    assert risk.sgd_class == SGD

def test_risk_initialization_with_noisygd(risk_setup):
    model, schedules, schedules_names, x0 = risk_setup
    risk = RiskComputations(model, x0, schedules, schedules_names, sgd_class=NoisyGD)
    
    assert risk.sgd_class == NoisyGD

def test_risk_default_schedules_names(risk_setup):
    model, schedules, _, x0 = risk_setup
    risk = RiskComputations(model, x0, schedules, sgd_class=SGD)
    
    # Si schedules_names n'est pas fourni, doit utiliser le nom des schedules
    assert len(risk.schedules_names) == len(schedules)
    for name in risk.schedules_names:
        assert isinstance(name, str)

def test_compute_all_theoretical_risks_dict_format(risk_setup):
    model, schedules, schedules_names, x0 = risk_setup
    risk = RiskComputations(model, x0, schedules, schedules_names, sgd_class=SGD)
    
    all_risks = risk.compute_all_theoretical_risks()
    
    assert isinstance(all_risks, dict)
    assert "Sched 1" in all_risks
    assert len(all_risks["Sched 1"]) == 5 # Nombre de steps dans le DummySchedule

def test_compute_risk_single_schedule(risk_setup_with_seed):
    model, schedules, schedules_names, x0 = risk_setup_with_seed
    risk = RiskComputations(model, x0, schedules, schedules_names, sgd_class=SGD)
    
    risk_single = risk.compute_risk(name="Sched 1")
    
    assert isinstance(risk_single, np.ndarray)
    assert len(risk_single) == 10
    assert np.all(risk_single >= 0)

def test_compute_all_empirical_risks(risk_setup_with_seed):
    model, schedules, schedules_names, x0 = risk_setup_with_seed
    risk = RiskComputations(model, x0, schedules, schedules_names, sgd_class=SGD)
    
    all_empirical_risks = risk.compute_all_empirical_risks(n_runs=3)
    
    assert isinstance(all_empirical_risks, dict)
    assert "Sched 1" in all_empirical_risks
    assert "Sched 2" in all_empirical_risks
    assert len(all_empirical_risks["Sched 1"]) == 10
    assert len(all_empirical_risks["Sched 2"]) == 10

def test_compute_mean_empirical_risk(risk_setup_with_seed):
    model, schedules, schedules_names, x0 = risk_setup_with_seed
    risk = RiskComputations(model, x0, schedules, schedules_names, sgd_class=SGD)
    
    mean_risk = risk.compute_mean_empirical_risk(n_runs=2, name="Sched 1")
    
    assert isinstance(mean_risk, np.ndarray)
    assert len(mean_risk) == 10
    assert np.all(mean_risk >= 0)

def test_compute_approx_vs_theoretical_risks(risk_setup_with_seed):
    model, schedules, schedules_names, x0 = risk_setup_with_seed
    risk = RiskComputations(model, x0, schedules, schedules_names, sgd_class=SGD)
    
    result = risk.compute_approx_vs_theoretical_risks()
    
    assert isinstance(result, dict)
    assert "theoretical" in result
    assert "approximate" in result
    assert isinstance(result["theoretical"], dict)
    assert isinstance(result["approximate"], dict)
    assert "Sched 1" in result["theoretical"]
    assert "Sched 1" in result["approximate"]
    assert len(result["theoretical"]["Sched 1"]) == len(result["approximate"]["Sched 1"])

def test_compute_all_risks(risk_setup_with_seed):
    model, schedules, schedules_names, x0 = risk_setup_with_seed
    risk = RiskComputations(model, x0, schedules, schedules_names, sgd_class=SGD)
    
    result = risk.compute_all_risks(n_runs=2)
    
    assert isinstance(result, dict)
    assert "theoretical" in result
    assert "empirical" in result
    assert isinstance(result["empirical"], dict)
    assert isinstance(result["theoretical"], dict)
    assert "Sched 1" in result["empirical"]
    assert "Sched 1" in result["theoretical"]
    assert len(result["empirical"]["Sched 1"]) == len(result["theoretical"]["Sched 1"])
    assert np.all(result["empirical"]["Sched 1"] >= 0)
    assert np.all(result["theoretical"]["Sched 1"] >= 0)

def test_approx_all_theoretical_risks(risk_setup_with_seed):
    model, schedules, schedules_names, x0 = risk_setup_with_seed
    risk = RiskComputations(model, x0, schedules, schedules_names, sgd_class=SGD)
    
    approx_risks = risk.approx_all_theoretical_risks()
    
    assert isinstance(approx_risks, dict)
    assert "Sched 1" in approx_risks
    assert "Sched 2" in approx_risks
    assert len(approx_risks["Sched 1"]) == 10

def test_optimize_base_lr(risk_setup_with_seed):
    model, schedules, schedules_names, x0 = risk_setup_with_seed
    risk = RiskComputations(model, x0, schedules, schedules_names, sgd_class=SGD)
    
    # Utiliser une petite plage d'etas pour tester rapidement
    eta_range = np.array([0.001, 0.01, 0.1])
    best_eta, min_risk = risk.optimize_base_lr(name="Sched 1", eta_range=eta_range, change_eta=False)
    
    # Le meilleur eta doit être dans la plage fournie
    assert best_eta in eta_range
    # Le risque final doit être positif
    assert min_risk >= 0
    # Le learning rate de la schedule ne doit pas avoir changé après l'optimisation
    # (puisque change_eta=False)
    original_lr = schedules[0].get_base_lr()
    assert np.allclose(risk.schedules["Sched 1"].get_base_lr(), original_lr)

def test_optimize_base_lr_with_change(risk_setup_with_seed):
    model, schedules, schedules_names, x0 = risk_setup_with_seed
    risk = RiskComputations(model, x0, schedules, schedules_names, sgd_class=SGD)
    
    eta_range = np.array([0.001, 0.01, 0.1])
    original_lr = schedules[0].get_base_lr()
    best_eta, min_risk = risk.optimize_base_lr(name="Sched 1", eta_range=eta_range, change_eta=True)
    
    # Après change_eta=True, le learning rate doit avoir été mis à jour au meilleur
    new_lr = risk.schedules["Sched 1"].get_base_lr()
    assert np.allclose(new_lr, best_eta)

def test_optimize_all_base_lrs(risk_setup_with_seed):
    model, schedules, schedules_names, x0 = risk_setup_with_seed
    risk = RiskComputations(model, x0, schedules, schedules_names, sgd_class=SGD)
    
    eta_range = np.array([0.001, 0.01, 0.1])
    results = risk.optimize_all_base_lrs(eta_range=eta_range, change_eta=False, save_results=False)
    
    assert isinstance(results, dict)
    assert "Sched 1" in results
    assert "Sched 2" in results
    assert "best_eta" in results["Sched 1"]
    assert "min_risk" in results["Sched 1"]
    assert results["Sched 1"]["best_eta"] in eta_range
    assert results["Sched 1"]["min_risk"] >= 0

def test_optimize_at_several_ts(risk_setup_with_seed):
    model, schedules, schedules_names, x0 = risk_setup_with_seed
    risk = RiskComputations(model, x0, schedules, schedules_names, sgd_class=SGD)
    
    t_values = [5, 9]
    eta_range = np.array([0.001, 0.01, 0.1])
    results = risk.optimize_at_several_ts(t_values=t_values, eta_range=eta_range, change_eta=False, save_results=False)
    
    assert isinstance(results, dict)
    for t in t_values:
        assert t in results
        assert "Sched 1" in results[t]
        assert "Sched 2" in results[t]
        assert "best_eta" in results[t]["Sched 1"]
        assert "min_risk" in results[t]["Sched 1"]
        assert results[t]["Sched 1"]["best_eta"] in eta_range
        assert results[t]["Sched 1"]["min_risk"] >= 0