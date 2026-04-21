import pytest
import numpy as np
from src.new_schedules.polynomial import PolynomialSchedule


@pytest.fixture
def polynomial_schedule_default():
    """Fixture to instantiate a standard PolynomialSchedule."""
    return PolynomialSchedule(steps=100, base_lr=0.1, exponent=0.5)


def test_polynomial_schedule_initialization(polynomial_schedule_default):
    """Test correct initialization of PolynomialSchedule."""
    assert polynomial_schedule_default._steps == 100
    assert polynomial_schedule_default._base_lr == 0.1
    assert polynomial_schedule_default._exponent == 0.5


def test_polynomial_schedule_name():
    """Test that the schedule name is correct."""
    schedule = PolynomialSchedule(steps=50, base_lr=0.05, exponent=1.0)
    assert schedule.name == 'polynomial'


def test_polynomial_schedule_construction():
    """Test that the schedule is built correctly."""
    schedule = PolynomialSchedule(steps=10, base_lr=1.0, exponent=0.5)
    
    # The schedule must have 10 elements (steps)
    assert len(schedule.schedule) == 10
    
    # Elements must be positive
    assert np.all(schedule.schedule > 0)


def test_polynomial_schedule_decay():
    """Test that the schedule decays correctly according to a power law."""
    steps = 100
    exponent = 0.5
    base_lr = 1.0
    schedule = PolynomialSchedule(steps=steps, base_lr=base_lr, exponent=exponent)
    
    # Verify that the schedule is decreasing
    schedule_arr = schedule.schedule
    for i in range(len(schedule_arr) - 1):
        assert schedule_arr[i] > schedule_arr[i + 1], f"Schedule is not decreasing at index {i}"


def test_polynomial_schedule_different_exponents():
    """Test that different exponents give different schedules."""
    steps = 50
    base_lr = 1.0
    
    schedule_exp_0_5 = PolynomialSchedule(steps=steps, base_lr=base_lr, exponent=0.5)
    schedule_exp_1_0 = PolynomialSchedule(steps=steps, base_lr=base_lr, exponent=1.0)
    schedule_exp_2_0 = PolynomialSchedule(steps=steps, base_lr=base_lr, exponent=2.0)
    
    # With larger exponents, the decay must be faster
    # So at fixed t, schedule_exp_0_5[t] > schedule_exp_1_0[t] > schedule_exp_2_0[t]
    mid_point = steps // 2
    assert schedule_exp_0_5.schedule[mid_point] > schedule_exp_1_0.schedule[mid_point]
    assert schedule_exp_1_0.schedule[mid_point] > schedule_exp_2_0.schedule[mid_point]


def test_polynomial_schedule_mathematical_formula():
    """Test that the schedule follows the formula (t+1)^(-exponent)."""
    steps = 20
    exponent = 0.5
    base_lr = 1.0
    schedule = PolynomialSchedule(steps=steps, base_lr=base_lr, exponent=exponent)
    
    # Compute the expected schedule
    expected = np.array([(t + 1)**(-exponent) for t in range(steps)])
    
    # Compare with the computed schedule (before potential base_lr application)
    # Note: the schedule may be multiplied by base_lr
    actual = np.array(schedule.schedule)
    
    # Verify that ratios are correct (eliminate the base_lr factor)
    ratio = actual[0] / expected[0]
    expected_scaled = expected * ratio
    
    assert np.allclose(actual, expected_scaled, rtol=1e-10)


def test_polynomial_schedule_base_lr_effect(polynomial_schedule_default):
    """Test that base_lr correctly affects the schedule."""
    schedule_high_lr = PolynomialSchedule(steps=50, base_lr=1.0, exponent=0.5)
    schedule_low_lr = PolynomialSchedule(steps=50, base_lr=0.1, exponent=0.5)
    
    # The ratios between schedules must be constant
    ratio = schedule_high_lr.schedule[0] / schedule_low_lr.schedule[0]
    
    for i in range(len(schedule_high_lr.schedule)):
        assert np.isclose(schedule_high_lr.schedule[i] / schedule_low_lr.schedule[i], ratio)


def test_polynomial_schedule_positive_always():
    """Test that all elements of the schedule are positive."""
    for exponent in [0.1, 0.5, 1.0, 2.0]:
        schedule = PolynomialSchedule(steps=100, base_lr=0.1, exponent=exponent)
        assert np.all(schedule.schedule > 0), f"Schedule a des valeurs non-positives pour exponent={exponent}"
