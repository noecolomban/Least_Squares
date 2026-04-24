from src.risk_computations import RiskComputations
from src.least_squares import PowerLawRegression
from src.SGD import SGD
import numpy as np
from scheduled import WSDSchedule, ConstantSchedule
from scipy.special import gamma


class AsymptoticsAnalysis:
    def __init__(self, model: PowerLawRegression, x0):
        # Removed T from init to emphasize it's independent from the base mathematical setup
        self.model = model
        self.x0 = x0
        self.m0 = self._compute_m0()

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


class Laplace_constant(AsymptoticsAnalysis):
    def __init__(self, model: PowerLawRegression, x0):
        # Initialize without a fixed T
        super().__init__(model, x0)
        self.T = None
        self.schedule = None
        self.sgd = None
        self.computations = None

    def _setup_for_T(self, T):
        """Configure the schedule and computations for a specific horizon T."""
        self.T = T
        self.schedule = ConstantSchedule(steps=T, base_lr=0.1)
        self.sgd = SGD(self.model, self.x0, self.schedule)
        self.computations = RiskComputations(
            self.model, self.x0, [self.schedule], ["constant"], sgd_class=SGD
        )
        
        # Optimize learning rate specifically for this T
        self.computations.optimize_all_base_lrs(t_value=T-1, change_eta=True)

    def compute_true_approx_bias(self, t):
        """Compute the bias term using the approximation."""
        L = self.model.Lambda_vals
        eta = self.schedule.get_base_lr()
        a = self.get_a_vals(eta)
        
        return 0.5 * np.sum(L * (a**t) * self.m0)

    def compute_true_approx_variance(self, t):
        """Compute the variance term using a geometric series."""
        eta = self.schedule.get_base_lr()
        L = self.model.Lambda_vals
        sigma_sq = self.model.sigma**2

        term = (L**2) * (eta**2) * sigma_sq
        a = self.get_a_vals(eta)
        
        # Apply geometric sum formula: sum(a^k) = (1 - a^t) / (1 - a)
        geom_sum = np.where(a == 1, t, (1 - a**t) / (1 - a))
            
        return 0.5 * np.sum(term * geom_sum)

    def compute_true_approx_risk_for_T(self, T):
        """Setup for T, optimize eta, and compute total true approximate risk."""
        self._setup_for_T(T)
        
        # Usually evaluated at T or T-1 depending on your exact convention
        # Here we use T directly as it seems to be the target evaluation step
        bias = self.compute_true_approx_bias(T)
        variance = self.compute_true_approx_variance(T)
        
        return bias + variance

    def compute_lagrange_approx_risk_for_T(self, T, m_exponent, m_constant):
        """Setup for T, optimize eta, and compute Lagrange approximate risk."""
        self._setup_for_T(T)
        eta = self.schedule.get_base_lr()
        sigma_sq = self.model.sigma**2
        
        # Bias
        C = (m_exponent - 1) / self.model.exponent + 1
        bias = m_constant / (2 * self.model.exponent) * gamma(C) / (2 * eta * T)**C
        
        # Variance
        variance = eta * sigma_sq / (2 * (self.model.exponent - 1))
        
        return bias + variance

#end of class definitions



def compute_real_approx_for_several_ts(T_values, model, x0):
    """Compute True risk approximation efficiently by sharing m0 across T."""
    results = {}
    
    # Instantiate Laplace analysis ONCE. m0 is computed here.
    print("Initializing Asymptotics Analysis and computing m0...")
    laplace_analysis = Laplace_constant(model, x0)
    
    for T in T_values:
        print(f"Optimizing and computing True risk approximation for T={T}...")
        # For each T, it will just re-init the schedule, run optimization, and compute
        risk = laplace_analysis.compute_true_approx_risk_for_T(T)
        results[T] = risk
        print(f"True risk approximation for T={T} computed: {risk}")
        
    return results


def compute_laplace_for_several_ts(T_values, model, x0, Delta, beta):
    """Compute Lagrange risk approximation efficiently by sharing m0 across T."""
    results = {}
    
    # Instantiate Laplace analysis ONCE. m0 is computed here.
    print("Initializing Asymptotics Analysis and computing m0...")
    laplace_analysis = Laplace_constant(model, x0)
    
    for T in T_values:
        print(f"Optimizing and computing Lagrange risk approximation for T={T}...")
        risk = laplace_analysis.compute_lagrange_approx_risk_for_T(
            T, m_exponent=beta, m_constant=Delta
        )
        results[T] = risk
        print(f"Laplace risk approximation for T={T} computed: {risk}")
        
    return results