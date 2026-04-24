from src.risk_computations import RiskComputations
from src.least_squares import PowerLawRegression
from src.SGD import SGD
import numpy as np
from scheduled import ConstantSchedule
from scipy.special import gamma
import concurrent.futures


class AsymptoticsAnalysis:
    def __init__(self, model: PowerLawRegression, x0, T=1000):
        self.model = model
        self.T = T
        self.x0 = x0
        self.m0 = self._compute_m0()

    def _compute_m0(self):
        """Compute m0 = diag(Q^T * (x0 - x*) * (x0 - x*)^T * Q)"""
        diff = self.x0.flatten() - self.model.x_star.flatten()
        Sigma0 = np.outer(diff, diff)
        _, m0 = self.model.compute_M_t(Sigma0)
        return m0

    @property
    def a_vals(self):
        """Compute all 'a' values simultaneously using vectorization."""
        eta = self.schedule.get_base_lr()
        L = self.model.Lambda_vals
        return (1 - eta * L)**2 + 2 * (eta**2) * (L**2)

    def a(self, i):
        """Compute the a_i term (approximation) for backward compatibility."""
        assert 0 <= i < self.model.dim, "i must be between 0 and dim-1"
        return self.a_vals[i]


class ZTransform_constant(AsymptoticsAnalysis):
    def __init__(self, model: PowerLawRegression, x0, T=1000):
        super().__init__(model, x0, T)
        self.schedule = ConstantSchedule(steps=T, base_lr=0.1)
        self.sgd = SGD(model, x0, self.schedule)
        self.computations = RiskComputations(model, x0, [self.schedule], ["constant"], sgd_class=SGD)

        self.computations.optimize_all_base_lrs(change_eta=True)
        self._z_transform_results = {}
        
    def compute_z_transform_result(self, i=None):
        """Compute the Z-transform result at step T using m0 and the eigenvalues."""
        if i is None:
            i = self.model.dim - 1
        assert 0 <= i < self.model.dim, "i must be between 0 and dim-1"
        
        if i in self._z_transform_results:
            return self._z_transform_results[i]

        Lambda_vals = self.model.Lambda_vals
        eta = self.schedule.get_base_lr()  
        sigma_sq = self.model.sigma**2

        if 0 not in self._z_transform_results:
            self._z_transform_results[0] = 0.5 * (eta**2 * Lambda_vals[0]**2 * sigma_sq) / (1 - self.a(0))

        current_max_t = max(self._z_transform_results.keys())
        
        for j in range(current_max_t + 1, i + 1):
            prev_val = self._z_transform_results[j - 1] 
            current_term = 0.5 * (eta**2 * Lambda_vals[j]**2 * sigma_sq) / (1 - self.a(j))
            self._z_transform_results[j] = prev_val + current_term

        return self._z_transform_results[i]

    def compute_all_approx_vs_z_transform(self):
        """Compute the Z-transform results and the approximate risks for all steps up to T-1.
        Affine interpolation for approx"""
        z_results_values = {"constant" : self.compute_z_transform_result() * np.ones(self.T)}
        approx_risks = self.computations.approx_all_theoretical_risks()
        
        return z_results_values, approx_risks
        

def z_transform_several_ts(list_T=None, sigma=0.01, dim=100, eta_range=None):
    """Compute and plot the Z-transform results at several time steps."""
    if list_T is None:
        list_T = [100, 500, 1000, 5000, 10000]
    
    results = {}
    for T in list_T:
        eta = 0.1
        if eta_range is None:
            eta_range = np.logspace(-4, 2, 30)

        model = PowerLawRegression(dim=dim, sigma=sigma, exponent=2)
        constant = ConstantSchedule(steps=T, base_lr=eta)

        beta = 0
        x0 = np.array([1 / i**beta for i in range(1, dim+1)])

        schedules1 = [constant]
        asymptotics_analysis = ZTransform_constant(model, x0, T=T)

        ztransform, approx = asymptotics_analysis.compute_all_approx_vs_z_transform()
        results[T] = (ztransform, approx)
    return results


class Laplace_constant(AsymptoticsAnalysis):
    def __init__(self, model: PowerLawRegression, x0, T=1000):
        super().__init__(model, x0, T)
        self.schedule = ConstantSchedule(steps=T, base_lr=0.1)
        self.sgd = SGD(model, x0, self.schedule)
        self.computations = RiskComputations(model, x0, [self.schedule], ["constant"], sgd_class=SGD)
        self.computations.optimize_all_base_lrs(change_eta=True)

    def compute_true_approx_bias(self, t=None):
        """Compute the bias term until step t using the approximation."""
        if t is None:
            t = self.T - 1
        assert 0 <= t < self.T, "t must be between 0 and T-1"
        
        L = self.model.Lambda_vals
        a = self.a_vals
        return 0.5 * np.sum(L * (a**t) * self.m0)
    
    def compute_lagrange_approx_bias(self, t=None, m_constant=None, m_exponent=None):
        """Compute the bias term at step t using the Lagrange method."""
        if t is None:
            t = self.T - 1
        if m_exponent is None or m_constant is None:
            raise ValueError("m_exponent and m_constant must be provided.")
        assert 0 <= t < self.T, "t must be between 0 and T-1"
        
        eta = self.schedule.get_base_lr()
        C = (m_exponent - 1) / self.model.exponent + 1
        return m_constant / (2 * self.model.exponent) * gamma(C) / (2 * eta * t)**C

    def compute_true_approx_variance(self, t=None):
        """Compute the variance term using a geometric series to avoid O(T) loops."""
        if t is None:
            t = self.T - 1
        assert 0 <= t < self.T, "t must be between 0 and T-1"

        eta = self.schedule.get_base_lr()
        L = self.model.Lambda_vals
        sigma_sq = self.model.sigma**2

        term = (L**2) * (eta**2) * sigma_sq
        a = self.a_vals
        
        # Apply geometric sum formula: sum(a^k) = (1 - a^t) / (1 - a)
        geom_sum = np.where(a == 1, t, (1 - a**t) / (1 - a))
            
        return 0.5 * np.sum(term * geom_sum)
    
    def compute_lagrange_approx_variance(self, t=None, m_exponent=None, m_constant=None):
        """Compute Lagrange approximate variance."""
        if t is None:
            t = self.T - 1
        if m_exponent is None or m_constant is None:
            raise ValueError("m_exponent and m_constant must be provided.")
        assert 0 <= t < self.T, "t must be between 0 and T-1"
        assert m_exponent > 1, "m_exponent must be greater than 1 unless the variance diverges"

        eta = self.schedule.get_base_lr()
        sigma_sq = self.model.sigma**2
        return eta * sigma_sq / (2 * (self.model.exponent - 1))
    
    def compute_true_approx_risk(self, t=None):
        """Compute total true approximate risk (bias + variance)."""
        return self.compute_true_approx_bias(t) + self.compute_true_approx_variance(t)

    def compute_lagrange_approx_risk(self, t=None, m_exponent=None, m_constant=None):
        """Compute total Lagrange approximate risk (bias + variance)."""
        bias = self.compute_lagrange_approx_bias(t, m_exponent=m_exponent, m_constant=m_constant)
        variance = self.compute_lagrange_approx_variance(t, m_exponent=m_exponent, m_constant=m_constant)
        return bias + variance


def _compute_single_laplace(T, model, x0, Delta, beta):
    """Helper function to isolate logic for a single T (useful for multiprocessing)."""
    laplace_analysis = Laplace_constant(model, x0, T=T)
    res = laplace_analysis.compute_lagrange_approx_risk(m_constant=Delta, m_exponent=beta)
    return T, res

def compute_laplace_for_several_ts(T_values, model, x0, Delta, beta):
    """Compute Laplace risk approximation concurrently for several T values."""
    results = {}
    print(f"Starting Laplace risk approximations for T_values: {T_values}...")
    
    # ProcessPoolExecutor allows bypassing the GIL for CPU-bound tasks
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(_compute_single_laplace, T, model, x0, Delta, beta) for T in T_values]
        
        for future in concurrent.futures.as_completed(futures):
            T, laplace_result = future.result()
            results[T] = laplace_result
            print(f"[Done] T={T} : {laplace_result}")
            
    return results

def _compute_single_real_approx(T, model, x0):
    """Helper function to isolate logic for a single T (useful for multiprocessing)."""
    laplace_analysis = Laplace_constant(model, x0, T=T)
    res = laplace_analysis.compute_true_approx_risk()
    return T, res

def compute_real_approx_for_several_ts(T_values, model, x0):
    """Compute True risk approximation concurrently for several T values."""
    results = {}
    print(f"Starting True risk approximations for T_values: {T_values}...")
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(_compute_single_real_approx, T, model, x0) for T in T_values]
        
        for future in concurrent.futures.as_completed(futures):
            T, laplace_result = future.result()
            results[T] = laplace_result
            print(f"[Done] T={T} : {laplace_result}")
            
    return results