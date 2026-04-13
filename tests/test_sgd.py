import pytest
import numpy as np
from src.least_squares import LinearRegression
from src.SGD import SGD, NoisyGD

# --- MOCK du Schedule ---
class DummySchedule:
    def __init__(self, steps=10, lr=0.01):
        self._steps = steps
        self.schedule = [lr] * steps
        self.name = "Dummy"

@pytest.fixture
def model_and_schedule():
    model = LinearRegression(dim=3, sigma=0.1, n_samples=50)
    schedule = DummySchedule(steps=10, lr=0.01)
    # x0 aléatoire
    x0 = np.random.randn(3, 1)
    return model, schedule, x0

def test_true_sgd_theoretical_risk(model_and_schedule):
    model, schedule, x0 = model_and_schedule
    sgd = SGD(model, x0, schedule)
    
    risks = sgd.compute_all_theoretical_risks()
    
    assert isinstance(risks, np.ndarray)
    assert len(risks) == schedule._steps
    # Le risque doit être positif
    assert np.all(risks >= 0)

def test_noisy_gd_theoretical_risk(model_and_schedule):
    model, schedule, x0 = model_and_schedule
    noisy_gd = NoisyGD(model, x0, schedule)
    
    risks = noisy_gd.compute_all_theoretical_risks()
    
    assert len(risks) == schedule._steps
    assert len(noisy_gd.risks) == schedule._steps