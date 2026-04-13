import pytest
import numpy as np
from src.least_squares import LinearRegression
from src.SGD import SGD
from src.compute import Computations
from tests.test_sgd import DummySchedule  # On réutilise notre mock

@pytest.fixture
def compute_setup():
    model = LinearRegression(dim=3)
    schedules = [DummySchedule(steps=5), DummySchedule(steps=5)]
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

def test_compute_all_theoretical_risks_dict_format(compute_setup):
    model, schedules, schedules_names = compute_setup
    comp = Computations(model, schedules, schedules_names, sgd_class=SGD)
    x0 = np.zeros(3)
    
    all_risks = comp.compute_all_theoretical_risks(x0, plot=False)
    
    assert isinstance(all_risks, dict)
    assert "Sched 1" in all_risks
    assert len(all_risks["Sched 1"]) == 5 # Nombre de steps dans le DummySchedule