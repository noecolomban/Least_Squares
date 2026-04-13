import pytest
import numpy as np
from src.new_schedules.polynomial import PolynomialSchedule


@pytest.fixture
def polynomial_schedule_default():
    """Fixture pour instancier un PolynomialSchedule standard."""
    return PolynomialSchedule(steps=100, base_lr=0.1, exponent=0.5)


def test_polynomial_schedule_initialization(polynomial_schedule_default):
    """Test l'initialisation correcte de PolynomialSchedule."""
    assert polynomial_schedule_default._steps == 100
    assert polynomial_schedule_default._base_lr == 0.1
    assert polynomial_schedule_default._exponent == 0.5


def test_polynomial_schedule_name():
    """Test que le nom du schedule est correct."""
    schedule = PolynomialSchedule(steps=50, base_lr=0.05, exponent=1.0)
    assert schedule.name == 'polynomial'


def test_polynomial_schedule_construction():
    """Test que la schedule est construite correctement."""
    schedule = PolynomialSchedule(steps=10, base_lr=1.0, exponent=0.5)
    
    # La schedule doit avoir 10 éléments (steps)
    assert len(schedule.schedule) == 10
    
    # Les éléments doivent être positifs
    assert np.all(schedule.schedule > 0)


def test_polynomial_schedule_decay():
    """Test que la schedule décroît correctement selon une loi de puissance."""
    steps = 100
    exponent = 0.5
    base_lr = 1.0
    schedule = PolynomialSchedule(steps=steps, base_lr=base_lr, exponent=exponent)
    
    # Vérifier que la schedule est décroissante
    schedule_arr = schedule.schedule
    for i in range(len(schedule_arr) - 1):
        assert schedule_arr[i] > schedule_arr[i + 1], f"Schedule n'est pas décroissante à l'index {i}"


def test_polynomial_schedule_different_exponents():
    """Test que différents exposants donnent des schedules différentes."""
    steps = 50
    base_lr = 1.0
    
    schedule_exp_0_5 = PolynomialSchedule(steps=steps, base_lr=base_lr, exponent=0.5)
    schedule_exp_1_0 = PolynomialSchedule(steps=steps, base_lr=base_lr, exponent=1.0)
    schedule_exp_2_0 = PolynomialSchedule(steps=steps, base_lr=base_lr, exponent=2.0)
    
    # Avec des exposants plus grands, la décroissance doit être plus rapide
    # Donc à t fixe, schedule_exp_0_5[t] > schedule_exp_1_0[t] > schedule_exp_2_0[t]
    mid_point = steps // 2
    assert schedule_exp_0_5.schedule[mid_point] > schedule_exp_1_0.schedule[mid_point]
    assert schedule_exp_1_0.schedule[mid_point] > schedule_exp_2_0.schedule[mid_point]


def test_polynomial_schedule_mathematical_formula():
    """Test que la schedule respecte la formule (t+1)^(-exponent)."""
    steps = 20
    exponent = 0.5
    base_lr = 1.0
    schedule = PolynomialSchedule(steps=steps, base_lr=base_lr, exponent=exponent)
    
    # Calculer la schedule attendue
    expected = np.array([(t + 1)**(-exponent) for t in range(steps)])
    
    # Comparer avec la schedule calculée (avant application du base_lr potentiellement)
    # Note: la schedule peut être multipliée par base_lr
    actual = np.array(schedule.schedule)
    
    # Vérifier que les rapports sont corrects (éliminer le facteur base_lr)
    ratio = actual[0] / expected[0]
    expected_scaled = expected * ratio
    
    assert np.allclose(actual, expected_scaled, rtol=1e-10)


def test_polynomial_schedule_base_lr_effect(polynomial_schedule_default):
    """Test que le base_lr affecte correctement la schedule."""
    schedule_high_lr = PolynomialSchedule(steps=50, base_lr=1.0, exponent=0.5)
    schedule_low_lr = PolynomialSchedule(steps=50, base_lr=0.1, exponent=0.5)
    
    # Les ratios entre les schedules doivent être constants
    ratio = schedule_high_lr.schedule[0] / schedule_low_lr.schedule[0]
    
    for i in range(len(schedule_high_lr.schedule)):
        assert np.isclose(schedule_high_lr.schedule[i] / schedule_low_lr.schedule[i], ratio)


def test_polynomial_schedule_positive_always():
    """Test que tous les éléments de la schedule sont positifs."""
    for exponent in [0.1, 0.5, 1.0, 2.0]:
        schedule = PolynomialSchedule(steps=100, base_lr=0.1, exponent=exponent)
        assert np.all(schedule.schedule > 0), f"Schedule a des valeurs non-positives pour exponent={exponent}"
