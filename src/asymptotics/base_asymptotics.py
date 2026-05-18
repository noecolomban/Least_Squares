from src.least_squares import PowerLawRegression
import numpy as np
from scipy.special import gamma, digamma
from abc import ABC, abstractmethod
from src.risk_computations import RiskComputations

def gamma_prime(x):
    """Compute the derivative of the gamma function using the digamma function."""
    return gamma(x) * digamma(x)


class AsymptoticsAnalysis(ABC):
    def __init__(self, model: PowerLawRegression, x0):
        self.model = model
        self.x0 = x0
        self.m0 = self._compute_m0()
        self.schedule = None
        self.sgd = None
        self.computations = None | RiskComputations

    @property
    def T(self):
        return self.schedule._steps

    def _compute_m0(self):
        """Compute m0 = diag(Q^T * (x0 - x*) * (x0 - x*)^T * Q)"""
        diff = self.x0.flatten() - self.model.x_star.flatten()
        Sigma0 = np.outer(diff, diff)
        _, m0 = self.model.compute_M_t(Sigma0)
        return m0

    def get_a_vals(self, eta):
        """Compute all 'a' values simultaneously using vectorization for a given eta."""
        L = self.model.Lambda_vals
        return (1 - eta * L)**2 + 2 * (eta**2) * (L**2)

    def compute_true_risks(self, T_values):
        """Compute true risks for different T values."""
        risks = {}
        for T in T_values:
            self._update_schedule_for_T(T)  # Update schedule for new T
            key_name = self.computations.schedules_names[0]  # there is only one schedule in our computations
            risks[T] = self.computations.compute_all_theoretical_risks()[key_name]
        return risks        
        
    def compute_true_approx_biases_and_variances(self, T_values, K=1):
        """Compute bias for different T values at t=K*T."""
        assert 0<= K <= 1, "K should be between 0 and 1 to ensure t=K*T is a valid step within the schedule."

        biases = {}
        variances = {}
        for T in T_values:
            self._update_schedule_for_T(T)  # Update schedule for new T
            list_bias, list_variance = self.sgd.approx_all_theoretical_risks(separate_bias_variance=True)
            t = int(K * (T-1))  
            biases[T] = list_bias[t]
            variances[T] = list_variance[t]
        return biases, variances
    

    def compute_true_approx_risks(self, T_values, K=1):
        """Compute True risk approximation for different T values at t=K*T."""
        assert 0<= K <= 1, "K should be between 0 and 1 to ensure t=K*T is a valid step within the schedule."
        biases, variances = self.compute_true_approx_biases_and_variances(T_values, K)
        risks = {T: biases[T] + variances[T] for T in T_values}
        return risks

    @abstractmethod
    def compute_laplace_approx_risk_for_T(self, T, m_exponent, m_constant):
        """Abstract method to compute Laplace risk approximation"""
        pass

    @abstractmethod
    def _update_schedule_for_T(self, T):
        """Abstract method to update the schedule for a new T"""
        pass

    def optimize_eta(self, m_constant, T, K=1, eta_min=0.001, eta_max=1.0, num_points=200):
        """Optimize eta for a specific T"""
        risks = {}
        for eta in np.linspace(eta_min, eta_max, num_points):
            self._update_schedule_for_T(T, new_eta=eta)  # Update schedule with new eta
            risks[eta] = self.compute_laplace_approx_risk_for_T(T, K*T, self.model.exponent, m_constant)  # Using m0[0] as a proxy for m_constant
        optimal_eta = min(risks, key=risks.get)
        return optimal_eta, risks[optimal_eta], risks

