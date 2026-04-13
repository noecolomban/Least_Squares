import pytest
import numpy as np
from src.least_squares import LinearRegression
from src.SGD import SGD, NoisyGD
from src.compute import Computations
from tests.test_sgd import DummySchedule  # On réutilise notre mock

@pytest.fixture
def compute_setup():
    model = LinearRegression(dim=3)
    schedules = [DummySchedule(steps=5), DummySchedule(steps=5)]
    schedules_names = ["Sched 1", "Sched 2"]
    return model, schedules, schedules_names

@pytest.fixture
def compute_setup_with_seed():
    """Fixture avec seed pour résultats reproductibles."""
    np.random.seed(42)
    model = LinearRegression(dim=3)
    schedules = [DummySchedule(steps=10), DummySchedule(steps=10)]
    schedules_names = ["Sched 1", "Sched 2"]
    return model, schedules, schedules_names

def test_computation_initialization_and_colors(compute_setup):
    model, schedules, schedules_names = compute_setup
    comp = Computations(model, schedules, schedules_names, sgd_class=SGD)
    
    assert len(comp.schedule_colors) == 2
    assert "Sched 1" in comp.schedule_colors
    assert "Sched 2" in comp.schedule_colors
    # Les couleurs doivent être différentes pour des schedules différents
    assert comp.schedule_colors["Sched 1"] != comp.schedule_colors["Sched 2"]

def test_computation_initialization_with_noisygd(compute_setup):
    model, schedules, schedules_names = compute_setup
    comp = Computations(model, schedules, schedules_names, sgd_class=NoisyGD)
    
    assert comp.sgd_class == NoisyGD
    assert comp.class_name == NoisyGD.name

def test_computation_default_schedules_names(compute_setup):
    model, schedules, _ = compute_setup
    comp = Computations(model, schedules, sgd_class=SGD)
    
    # Si schedules_names n'est pas fourni, doit utiliser le nom des schedules
    assert len(comp.schedules_names) == len(schedules)
    for name in comp.schedules_names:
        assert isinstance(name, str)

def test_compute_all_theoretical_risks_dict_format(compute_setup):
    model, schedules, schedules_names = compute_setup
    comp = Computations(model, schedules, schedules_names, sgd_class=SGD)
    x0 = np.zeros(3)
    
    all_risks = comp.compute_all_theoretical_risks(x0, plot=False)
    
    assert isinstance(all_risks, dict)
    assert "Sched 1" in all_risks
    assert len(all_risks["Sched 1"]) == 5 # Nombre de steps dans le DummySchedule

def test_compute_risk_single_schedule(compute_setup_with_seed):
    model, schedules, _ = compute_setup_with_seed
    comp = Computations(model, schedules, sgd_class=SGD)
    x0 = np.random.randn(3, 1)
    
    risk = comp.compute_risk(x0, i_schedule=0)
    
    assert isinstance(risk, np.ndarray)
    assert len(risk) == 10
    assert np.all(risk >= 0)

def test_compute_all_empirical_risks(compute_setup_with_seed):
    model, schedules, schedules_names = compute_setup_with_seed
    comp = Computations(model, schedules, schedules_names, sgd_class=SGD)
    x0 = np.random.randn(3, 1)
    
    all_empirical_risks = comp.compute_all_empirical_risks(x0, n_runs=3, plot=False)
    
    assert isinstance(all_empirical_risks, dict)
    assert "Sched 1" in all_empirical_risks
    assert "Sched 2" in all_empirical_risks
    assert len(all_empirical_risks["Sched 1"]) == 10
    assert len(all_empirical_risks["Sched 2"]) == 10

def test_compute_mean_empirical_risk(compute_setup_with_seed):
    model, schedules, _ = compute_setup_with_seed
    comp = Computations(model, schedules, sgd_class=SGD)
    x0 = np.random.randn(3, 1)
    
    mean_risk = comp.compute_mean_empirical_risk(x0, n_runs=2, i_schedule=0)
    
    assert isinstance(mean_risk, np.ndarray)
    assert len(mean_risk) == 10
    assert np.all(mean_risk >= 0)

def test_compute_approx_vs_theoretical_risks(compute_setup_with_seed):
    model, schedules, schedules_names = compute_setup_with_seed
    comp = Computations(model, schedules, schedules_names, sgd_class=SGD)
    x0 = np.random.randn(3, 1)
    
    theoretical, approx = comp.compute_approx_vs_theoretical_risks(x0, plot=False)
    
    assert isinstance(theoretical, dict)
    assert isinstance(approx, dict)
    assert "Sched 1" in theoretical
    assert "Sched 1" in approx
    assert len(theoretical["Sched 1"]) == len(approx["Sched 1"])

def test_compute_all_risks(compute_setup_with_seed):
    model, schedules, schedules_names = compute_setup_with_seed
    comp = Computations(model, schedules, schedules_names, sgd_class=SGD)
    x0 = np.random.randn(3, 1)
    
    empirical, theoretical = comp.compute_all_risks(x0, n_runs=2, plot=False)
    
    assert isinstance(empirical, dict)
    assert isinstance(theoretical, dict)
    assert "Sched 1" in empirical
    assert "Sched 1" in theoretical
    assert len(empirical["Sched 1"]) == len(theoretical["Sched 1"])
    assert np.all(empirical["Sched 1"] >= 0)
    assert np.all(theoretical["Sched 1"] >= 0)

def test_approx_all_theoretical_risks(compute_setup_with_seed):
    model, schedules, schedules_names = compute_setup_with_seed
    comp = Computations(model, schedules, schedules_names, sgd_class=SGD)
    x0 = np.random.randn(3, 1)
    
    approx_risks = comp.approx_all_theoretical_risks(x0, plot=False)
    
    assert isinstance(approx_risks, dict)
    assert "Sched 1" in approx_risks
    assert "Sched 2" in approx_risks
    assert len(approx_risks["Sched 1"]) == 10

def test_get_schedule_color(compute_setup):
    model, schedules, schedules_names = compute_setup
    comp = Computations(model, schedules, schedules_names, sgd_class=SGD)
    
    color = comp._get_schedule_color("Sched 1")
    
    # Les couleurs doivent être au format hex
    assert isinstance(color, str)
    assert color.startswith('#')
    assert len(color) == 7  # Format hex standard: #RRGGBB

def test_get_nonexistent_schedule_color(compute_setup):
    model, schedules, schedules_names = compute_setup
    comp = Computations(model, schedules, schedules_names, sgd_class=SGD)
    
    # Pour une schedule inexistante, doit retourner la couleur par défaut
    default_color = comp._get_schedule_color("NonExistent")
    assert default_color == '#7f7f7f'  # gris par défaut

def test_generate_schedule_colors(compute_setup):
    model, schedules, schedules_names = compute_setup
    comp = Computations(model, schedules, schedules_names, sgd_class=SGD)
    
    colors = comp._generate_schedule_colors()
    
    assert isinstance(colors, dict)
    assert len(colors) == 2
    # Les couleurs doivent être uniques pour différentes schedules
    color_values = list(colors.values())
    assert len(set(color_values)) == len(color_values)

def test_optimize_base_lr(compute_setup_with_seed):
    model, schedules, schedules_names = compute_setup_with_seed
    comp = Computations(model, schedules, schedules_names, sgd_class=SGD)
    x0 = np.random.randn(3, 1)
    
    # Utiliser une petite plage d'etas pour tester rapidement
    eta_range = np.array([0.001, 0.01, 0.1])
    best_eta, min_risk = comp.optimize_base_lr(x0, i_schedule=0, eta_range=eta_range, 
                                                plot=False, change_eta=False)
    
    # Le meilleur eta doit être dans la plage fournie
    assert best_eta in eta_range
    # Le risque final doit être positif
    assert min_risk >= 0
    # Le learning rate de la schedule ne doit pas avoir changé après l'optimisation
    # (puisque change_eta=False)
    original_lr = schedules[0].get_base_lr()
    assert np.allclose(original_lr, original_lr)

def test_optimize_base_lr_with_change(compute_setup_with_seed):
    model, schedules, schedules_names = compute_setup_with_seed
    comp = Computations(model, schedules, schedules_names, sgd_class=SGD)
    x0 = np.random.randn(3, 1)
    
    eta_range = np.array([0.001, 0.01, 0.1])
    original_lr = schedules[0].get_base_lr()
    best_eta, min_risk = comp.optimize_base_lr(x0, i_schedule=0, eta_range=eta_range, 
                                                plot=False, change_eta=True)
    
    # Après change_eta=True, le learning rate doit avoir été mis à jour au meilleur
    new_lr = schedules[0].get_base_lr()
    assert np.allclose(new_lr, best_eta)