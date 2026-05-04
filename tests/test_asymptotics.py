import pytest
import numpy as np
from scipy.special import gamma
from src.asymptotics import Laplace_constant, Laplace_linear
from src.least_squares import PowerLawRegression, compute_power_x0


@pytest.fixture(scope="module")
def small_model():
    """Small PowerLawRegression for fast tests."""
    np.random.seed(0)
    return PowerLawRegression(dim=10, sigma=0.1, exponent=2)


@pytest.fixture(scope="module")
def x0_fixture(small_model):
    """Simple x0 from compute_power_x0 (beta=1)."""
    return compute_power_x0(small_model.dim, small_model.x_star, small_model.Q, beta=1)


@pytest.fixture(scope="module")
def laplace_const(small_model, x0_fixture):
    return Laplace_constant(small_model, x0_fixture, T_max=50)


@pytest.fixture(scope="module")
def laplace_lin(small_model, x0_fixture):
    return Laplace_linear(small_model, x0_fixture, T_max=50)


# ---------------------------------------------------------------------------
# compute_power_x0
# ---------------------------------------------------------------------------

class TestComputePowerX0:
    def test_shape(self, small_model):
        x0 = compute_power_x0(small_model.dim, small_model.x_star, small_model.Q, beta=1)
        assert x0.flatten().shape == (small_model.dim,)

    def test_delta0_power_law(self, small_model):
        """delta0 = Q^T (x0 - x*) should equal [1/i^beta] in the eigenvector basis."""
        beta = 1
        x0 = compute_power_x0(small_model.dim, small_model.x_star, small_model.Q, beta=beta)
        delta0 = small_model.Q.T @ (x0.flatten() - small_model.x_star.flatten())
        expected = np.array([1.0 / (i**beta) for i in range(1, small_model.dim + 1)])
        assert np.allclose(delta0.flatten(), expected)

    def test_beta_zero_uniform(self, small_model):
        """beta=0 => delta0 is all ones."""
        x0 = compute_power_x0(small_model.dim, small_model.x_star, small_model.Q, beta=0)
        delta0 = small_model.Q.T @ (x0.flatten() - small_model.x_star.flatten())
        assert np.allclose(delta0.flatten(), np.ones(small_model.dim))


# ---------------------------------------------------------------------------
# AsymptoticsAnalysis base class (tested through Laplace_constant)
# ---------------------------------------------------------------------------

class TestAsymptoticsBase:
    def test_m0_shape(self, laplace_const):
        assert laplace_const.m0.shape == (laplace_const.model.dim,)

    def test_m0_nonnegative(self, laplace_const):
        """m0 are diagonal elements of a PSD matrix, so must be non-negative."""
        assert np.all(laplace_const.m0 >= 0)

    def test_get_a_vals_shape(self, laplace_const):
        eta = laplace_const.schedule.get_base_lr()
        a_vals = laplace_const.get_a_vals(eta)
        assert a_vals.shape == (laplace_const.model.dim,)

    def test_get_a_vals_formula(self, laplace_const):
        """a_i = (1 - eta*lambda_i)^2 + 2*eta^2*lambda_i^2"""
        eta = laplace_const.schedule.get_base_lr()
        L = laplace_const.model.Lambda_vals
        expected = (1 - eta * L) ** 2 + 2 * eta ** 2 * L ** 2
        assert np.allclose(laplace_const.get_a_vals(eta), expected)

    def test_compute_true_approx_risks_keys(self, laplace_const):
        T_values = [30, 40, 50]
        risks = laplace_const.compute_true_approx_risks(T_values, K=1)
        assert set(risks.keys()) == set(T_values)

    def test_compute_true_approx_risks_nonnegative(self, laplace_const):
        T_values = [30, 50]
        risks = laplace_const.compute_true_approx_risks(T_values, K=1)
        for v in risks.values():
            assert v >= 0

    def test_compute_true_approx_biases_and_variances_shapes(self, laplace_const):
        T_values = [30, 50]
        biases, variances = laplace_const.compute_true_approx_biases_and_variances(T_values, K=1)
        assert set(biases.keys()) == set(T_values)
        assert set(variances.keys()) == set(T_values)

    def test_compute_true_approx_risks_equals_bias_plus_variance(self, laplace_const):
        T_values = [40, 50]
        biases, variances = laplace_const.compute_true_approx_biases_and_variances(T_values, K=1)
        risks = laplace_const.compute_true_approx_risks(T_values, K=1)
        for T in T_values:
            assert np.isclose(risks[T], biases[T] + variances[T])

    def test_K_out_of_range(self, laplace_const):
        with pytest.raises(AssertionError):
            laplace_const.compute_true_approx_risks([50], K=1.5)
        with pytest.raises(AssertionError):
            laplace_const.compute_true_approx_risks([50], K=-0.1)


# ---------------------------------------------------------------------------
# Laplace_constant
# ---------------------------------------------------------------------------

class TestLaplaceConstant:
    def test_instantiation(self, laplace_const):
        assert laplace_const is not None

    def test_schedule_steps(self, laplace_const):
        assert laplace_const.schedule._steps == laplace_const.T

    def test_eta_positive(self, laplace_const):
        assert laplace_const.schedule.get_base_lr() > 0

    def test_compute_laplace_risk_positive(self, laplace_const):
        risk = laplace_const.compute_laplace_approx_risk_for_T(50, m_exponent=2, m_constant=1.0)
        assert risk > 0

    def test_compute_laplace_risk_formula(self, laplace_const):
        """Check that the risk equals bias + variance computed manually."""
        T = 50
        m_exponent = 2
        m_constant = 1.0
        eta = laplace_const.schedule.get_base_lr()
        alpha = laplace_const.model.exponent
        sigma_sq = laplace_const.model.sigma ** 2

        C = (m_exponent - 1) / alpha + 1
        expected_bias = m_constant / (2 * alpha) * gamma(C) / (2 * eta * T) ** C
        expected_variance = eta * sigma_sq / (2 * (alpha - 1))
        expected = expected_bias + expected_variance

        result = laplace_const.compute_laplace_approx_risk_for_T(T, m_exponent, m_constant)
        assert np.isclose(result, expected)

    def test_bias_decreases_with_T(self, laplace_const):
        """Bias term should decrease as T grows (holding eta fixed)."""
        r1 = laplace_const.compute_laplace_approx_risk_for_T(50, m_exponent=3, m_constant=1.0)
        r2 = laplace_const.compute_laplace_approx_risk_for_T(500, m_exponent=3, m_constant=1.0)
        # For large enough T, larger T means smaller bias
        assert r1 >= r2

    def test_update_schedule_for_T(self, laplace_const):
        """_update_schedule_for_T should update steps without changing base_lr."""
        eta_before = laplace_const.schedule.get_base_lr()
        laplace_const._update_schedule_for_T(80)
        assert laplace_const.schedule._steps == 80
        assert laplace_const.schedule.get_base_lr() == eta_before
        # Restore
        laplace_const._update_schedule_for_T(50)


# ---------------------------------------------------------------------------
# Laplace_linear
# ---------------------------------------------------------------------------

class TestLaplaceLinear:
    def test_instantiation(self, laplace_lin):
        assert laplace_lin is not None

    def test_eta_positive(self, laplace_lin):
        assert laplace_lin.schedule.get_base_lr() > 0

    def test_update_schedule_for_T(self, laplace_lin):
        """_update_schedule_for_T should update the schedule steps."""
        eta_before = laplace_lin.schedule.get_base_lr()
        laplace_lin._update_schedule_for_T(80)
        assert laplace_lin.schedule._steps == 80
        assert laplace_lin.schedule.get_base_lr() == eta_before
        # Restore
        laplace_lin._update_schedule_for_T(50)

    def test_bias_at_t0_is_infinite(self, laplace_lin):
        """At t=0, bias_base=0, so bias should be +inf."""
        bias = laplace_lin.compute_laplace_approx_bias(50, t=0, m_exponent=2, m_constant=1.0)
        assert bias == float('inf')

    def test_bias_positive(self, laplace_lin):
        bias = laplace_lin.compute_laplace_approx_bias(50, t=30, m_exponent=2, m_constant=1.0)
        assert bias > 0

    def test_variance_at_t0_is_zero(self, laplace_lin):
        variance = laplace_lin.compute_laplace_approx_variance(50, t=0)
        assert variance == 0.0

    def test_variance_positive_for_t_gt_1(self, laplace_lin):
        variance = laplace_lin.compute_laplace_approx_variance(50, t=5)
        assert variance > 0

    def test_risk_equals_bias_plus_variance(self, laplace_lin):
        T, t = 50, 30
        bias = laplace_lin.compute_laplace_approx_bias(T, t, m_exponent=2, m_constant=1.0)
        variance = laplace_lin.compute_laplace_approx_variance(T, t)
        risk = laplace_lin.compute_laplace_approx_risk_for_T(T, t, m_exponent=2, m_constant=1.0)
        assert np.isclose(risk, bias + variance)

    def test_bias_decreases_with_t(self, laplace_lin):
        """Bias should decrease as t grows within a fixed T."""
        T = 50
        b1 = laplace_lin.compute_laplace_approx_bias(T, t=20, m_exponent=2, m_constant=1.0)
        b2 = laplace_lin.compute_laplace_approx_bias(T, t=40, m_exponent=2, m_constant=1.0)
        assert b1 >= b2

    def test_biases_and_variances_different_finals_keys(self, laplace_lin):
        T_values = [30, 40, 50]
        biases, variances = laplace_lin.compute_laplace_approx_biases_and_variances_different_finals(
            T_values, m_exponent=2, m_constant=1.0, K=1
        )
        assert set(biases.keys()) == set(T_values)
        assert set(variances.keys()) == set(T_values)

    def test_biases_and_variances_different_finals_nonneg(self, laplace_lin):
        T_values = [30, 50]
        biases, variances = laplace_lin.compute_laplace_approx_biases_and_variances_different_finals(
            T_values, m_exponent=2, m_constant=1.0, K=0.5
        )
        for T in T_values:
            assert biases[T] >= 0
            assert variances[T] >= 0

    def test_biases_and_variances_K_out_of_range(self, laplace_lin):
        with pytest.raises(AssertionError):
            laplace_lin.compute_laplace_approx_biases_and_variances_different_finals(
                [50], m_exponent=2, m_constant=1.0, K=1.5
            )
