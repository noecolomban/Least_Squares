import pytest
import numpy as np
from src.asymptotics import ZTransform_constant
from src.least_squares import PowerLawRegression, compute_power_x0


@pytest.fixture
def small_model():
    """Small PowerLawRegression for fast tests."""
    np.random.seed(0)
    return PowerLawRegression(dim=10, sigma=0.01, exponent=2)


@pytest.fixture
def x0_fixture(small_model):
    """Simple x0 from compute_power_x0 (beta=0 => uniform delta0)."""
    return compute_power_x0(small_model.dim, small_model.x_star, small_model.Q, beta=0)


@pytest.fixture
def ztransform(small_model, x0_fixture):
    return ZTransform_constant(small_model, x0_fixture, T=50)


# ---------------------------------------------------------------------------
# compute_power_x0
# ---------------------------------------------------------------------------

class TestComputePowerX0:
    def test_shape(self, small_model):
        x0 = compute_power_x0(small_model.dim, small_model.x_star, small_model.Q, beta=1)
        assert x0.shape == small_model.x_star.shape

    def test_delta0_power_law(self, small_model):
        """delta0 = Q^T (x0 - x*) should equal [1/i^beta] in the eigenvector basis."""
        beta = 1
        x0 = compute_power_x0(small_model.dim, small_model.x_star, small_model.Q, beta=beta)
        delta0 = small_model.Q.T @ (x0 - small_model.x_star)
        expected = np.array([1.0 / (i**beta) for i in range(1, small_model.dim + 1)])
        assert np.allclose(delta0.flatten(), expected)

    def test_beta_zero_uniform(self, small_model):
        """beta=0 => delta0 is all ones."""
        x0 = compute_power_x0(small_model.dim, small_model.x_star, small_model.Q, beta=0)
        delta0 = small_model.Q.T @ (x0 - small_model.x_star)
        assert np.allclose(delta0.flatten(), np.ones(small_model.dim))


# ---------------------------------------------------------------------------
# ZTransform_constant initialisation
# ---------------------------------------------------------------------------

class TestZTransformInit:
    def test_m0_shape(self, ztransform):
        assert ztransform.m0.shape == (ztransform.model.dim,)

    def test_m0_nonnegative(self, ztransform):
        """m0 are diagonal elements of a PSD matrix, so must be non-negative."""
        assert np.all(ztransform.m0 >= 0)

    def test_schedule_steps(self, ztransform):
        assert ztransform.schedule._steps == ztransform.T

    def test_eta_positive(self, ztransform):
        assert ztransform.schedule.get_base_lr() > 0


# ---------------------------------------------------------------------------
# a(i) coefficient
# ---------------------------------------------------------------------------

class TestACoefficient:
    def test_value(self, ztransform):
        """a(i) = (1 - eta*lambda_i)^2 + 2*eta^2*lambda_i^2"""
        eta = ztransform.schedule.get_base_lr()
        for i in range(ztransform.model.dim):
            lam = ztransform.model.Lambda_vals[i]
            expected = (1 - eta * lam) ** 2 + 2 * eta ** 2 * lam ** 2
            assert np.isclose(ztransform.a(i), expected)

    def test_cache(self, ztransform):
        """Second call should return cached value (same object)."""
        val1 = ztransform.a(0)
        val2 = ztransform.a(0)
        assert val1 == val2

    def test_out_of_bounds(self, ztransform):
        with pytest.raises(AssertionError):
            ztransform.a(-1)
        with pytest.raises(AssertionError):
            ztransform.a(ztransform.model.dim)


# ---------------------------------------------------------------------------
# compute_z_transform_result
# ---------------------------------------------------------------------------

class TestZTransformResult:
    def test_returns_float(self, ztransform):
        result = ztransform.compute_z_transform_result()
        assert isinstance(result, float) or np.isscalar(result)

    def test_nonnegative(self, ztransform):
        assert ztransform.compute_z_transform_result() >= 0

    def test_monotone_in_i(self, ztransform):
        """Each extra eigenvalue adds a non-negative term, so the result is non-decreasing in i."""
        results = [ztransform.compute_z_transform_result(i) for i in range(ztransform.model.dim)]
        diffs = np.diff(results)
        assert np.all(diffs >= -1e-12)

    def test_index_zero_formula(self, ztransform):
        """The i=0 term equals 0.5 * eta^2 * lambda_0^2 * sigma^2 / (1 - a(0))."""
        eta = ztransform.schedule.get_base_lr()
        lam0 = ztransform.model.Lambda_vals[0]
        sigma_sq = ztransform.model.sigma ** 2
        expected = 0.5 * (eta ** 2 * lam0 ** 2 * sigma_sq) / (1 - ztransform.a(0))
        assert np.isclose(ztransform.compute_z_transform_result(0), expected)

    def test_cache(self, ztransform):
        v1 = ztransform.compute_z_transform_result(3)
        v2 = ztransform.compute_z_transform_result(3)
        assert v1 == v2

    def test_out_of_bounds(self, ztransform):
        with pytest.raises(AssertionError):
            ztransform.compute_z_transform_result(-1)
        with pytest.raises(AssertionError):
            ztransform.compute_z_transform_result(ztransform.model.dim)


# ---------------------------------------------------------------------------
# compute_all_approx_vs_z_transform
# ---------------------------------------------------------------------------

class TestComputeAllApproxVsZTransform:
    def test_returns_two_dicts(self, ztransform):
        z_results, approx_risks = ztransform.compute_all_approx_vs_z_transform()
        assert isinstance(z_results, dict)
        assert isinstance(approx_risks, dict)

    def test_constant_key_present(self, ztransform):
        z_results, approx_risks = ztransform.compute_all_approx_vs_z_transform()
        assert "constant" in z_results
        assert "constant" in approx_risks

    def test_z_results_constant_value(self, ztransform):
        """Z-transform result is constant over time (same value repeated T times)."""
        z_results, _ = ztransform.compute_all_approx_vs_z_transform()
        arr = z_results["constant"]
        assert arr.shape == (ztransform.T,)
        assert np.all(arr == arr[0])

    def test_approx_risks_length(self, ztransform):
        _, approx_risks = ztransform.compute_all_approx_vs_z_transform()
        assert len(approx_risks["constant"]) == ztransform.T

    def test_approx_risks_nonnegative(self, ztransform):
        _, approx_risks = ztransform.compute_all_approx_vs_z_transform()
        assert np.all(approx_risks["constant"] >= 0)
