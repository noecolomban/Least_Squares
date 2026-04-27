from matplotlib.pylab import beta

from src.risk_computations import RiskComputations
from src.least_squares import PowerLawRegression
from src.SGD import SGD
import numpy as np
from scheduled import WSDSchedule, ConstantSchedule
from scipy.special import gamma
from abc import ABC, abstractmethod


class AsymptoticsAnalysis(ABC):
    def __init__(self, model: PowerLawRegression, x0):
        # Removed T from init to emphasize it's independent from the base mathematical setup
        self.model = model
        self.x0 = x0
        self.m0 = self._compute_m0()
        self.schedule = None
        self.sgd = None
        self.computations = None
        self.T = None


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
        
        # Usually evaluated at T or T-1 depending on your exact convention
        # Here we use T directly as it seems to be the target evaluation step
        bias = self.compute_true_approx_bias(T)
        variance = self.compute_true_approx_variance(T)
        
        return bias + variance

    def compute_real_approx_for_several_ts(self, T, step=10):
        """Compute True risk approximation efficiently by sharing m0 across T."""
        results = {}
        
        print("Computing True approximation...")
        for t in range(1, T+1, step):
            if t % (step * 10) == 1:  # Print progress every 10 steps
                print(f"Computing for t={t}...")
            risk = self.compute_true_approx_risk_for_T(t)
            results[t] = risk
        print(f"True risk approximation for t={t} computed: {risk}")
            
        return results

    def compute_laplace_for_several_ts(self, T, Delta, beta, step=10):
        """Compute Lagrange risk approximation efficiently by sharing m0 across T."""
        results = {}
        print("Computing Laplace risk approximation...")
        for t in range(1, T+1, step):
            if t % (step * 10) == 1:  # Print progress every 10 steps
                print(f"Computing for T={t}...")
            risk = self.compute_laplace_approx_risk_for_T(
                t, m_exponent=beta, m_constant=Delta
            )
            results[t] = risk
        print(f"Laplace risk approximation for T={T} computed: {risk}")
            
        return results
    
    @abstractmethod
    def compute_laplace_approx_risk_for_T(self, T, m_exponent, m_constant):
        """Abstract method to compute Laplace risk approximation"""
        pass


class Laplace_constant(AsymptoticsAnalysis):
    def __init__(self, model: PowerLawRegression, x0, T_max=100000):
        # Initialize without a fixed T
        super().__init__(model, x0)
        print(f"Initializing Laplace_constant with T_max={T_max} for setup...")
        self._setup_for_T(T_max)  # Setup for a large T to compute m0 and optimize eta


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


    def compute_laplace_approx_risk_for_T(self, T, m_exponent, m_constant):
        """Setup for T, optimize eta, and compute Lagrange approximate risk."""
        eta = self.schedule.get_base_lr()
        sigma_sq = self.model.sigma**2
        
        # Bias
        C = (m_exponent - 1) / self.model.exponent + 1
        bias = m_constant / (2 * self.model.exponent) * gamma(C) / (2 * eta * T)**C
        
        # Variance
        variance = eta * sigma_sq / (2 * (self.model.exponent - 1))
        
        return bias + variance



class Laplace_linear(AsymptoticsAnalysis):
    def __init__(self, model: PowerLawRegression, x0, T_max=100000):
        super().__init__(model, x0)
        print(f"Initializing Laplace_linear with T_max={T_max} for setup...")
        self._setup_for_T(T_max)


    def _setup_for_T(self, T):
        """Configure the schedule and computations for a specific horizon T."""
        self.T = T
        self.schedule = WSDSchedule(steps=T, base_lr=0.1, cooldown_len=1.)
        self.sgd = SGD(self.model, self.x0, self.schedule)
        self.computations = RiskComputations(
            self.model, self.x0, [self.schedule], ["linear"], sgd_class=SGD
        )
        
        # Optimize learning rate specifically for this T
        self.computations.optimize_all_base_lrs(t_value=T-1, change_eta=True)


    def compute_laplace_approx_risk_for_T(self, T, m_exponent, m_constant):
        """Setup for T, optimize eta, and compute 2nd-order Laplace approximate risk for a linear schedule."""
        # Note: Assuming self._setup_for_T(T) is called outside or not needed in this specific snippet
        eta = self.schedule.get_base_lr() 
        sigma_sq = self.model.sigma**2
        alpha = self.model.exponent
        
        C1 = (m_exponent - 1) / alpha + 1
        # C2 corresponds to (beta - 1) / alpha + 3 (for the 2nd order correction)
        C2 = C1 + 2 
        
        bias_base = eta * (T + 1)
        
        # Calculate the sum of squares for the bias correction: sum_{j=0}^{T-1} (eta * (T-j) / T)^2
        # Using the arithmetic sum of squares formula: sum_{i=1}^{T} i^2 = T(T+1)(2T+1)/6
        bias_sum_sq = (eta**2 / T**2) * (T * (T + 1) * (2 * T + 1)) / 6
        
        # 1st order and 2nd order bias components
        bias_1st_order = (m_constant / (2 * alpha)) * gamma(C1) / (bias_base ** C1)
        bias_2nd_order = (m_constant / (2 * alpha)) * bias_sum_sq * gamma(C2) / (bias_base ** C2)
        
        bias = bias_1st_order + bias_2nd_order
        
        
       
        Cv1 = (2 * alpha - 1) / alpha
        Cv2 = Cv1 + 2
        
        variance_prefix = (sigma_sq * eta**2) / (2 * alpha)
        
        k = np.arange(T - 1)
        n = T - k - 1 
        
        var_base = (eta / T) * n * (T - k)
        
        # Calculate the sum of squares for the variance correction: sum_{j=k+1}^{T-1} (eta * (T-j) / T)^2
        # Using the arithmetic sum of squares formula: sum_{i=1}^{n} i^2 = n(n+1)(2n+1)/6
        var_sum_sq = (eta**2 / T**2) * (n * (n + 1) * (2 * n + 1)) / 6
        
        # 1st order variance terms
        terms_1st = ((T - k) / T)**2 / (var_base ** Cv1)
        sum_terms_1st = np.sum(terms_1st) * gamma(Cv1)
        
        # 2nd order variance correction terms
        terms_2nd = ((T - k) / T)**2 * var_sum_sq / (var_base ** Cv2)
        sum_terms_2nd = np.sum(terms_2nd) * gamma(Cv2)
        
        # Special case for the last term (k = T - 1) where var_base becomes 0
        last_term = (1 / T)**2 * (alpha / (2 * alpha - 1))
        
        variance = variance_prefix * (sum_terms_1st + sum_terms_2nd + last_term)
        
        return bias + variance


#End of class definitions

def compute_different_sigmas(T, model, x0, Delta, beta, sigmas, schedule_type="constant"):
    """Compute Laplace risk approximation for different noise levels."""
    results = {}
    real_approx = {}
    
    for sigma in sigmas:
        print(f"Computing for sigma={sigma}...")
        model.sigma = sigma  # Update the noise level in the model
        analysis = Laplace_constant(model, x0, T) if schedule_type == "constant" else Laplace_linear(model, x0, T)
        risk = analysis.compute_laplace_for_several_ts(T, Delta, beta)
        real_approx[sigma] = analysis.compute_real_approx_for_several_ts(T)
        results[sigma] = risk
        print(f"Laplace risk approximation for sigma={sigma} computed.")
        
    return results, real_approx