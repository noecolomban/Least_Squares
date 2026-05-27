from .base_asymptotics import AsymptoticsAnalysis, gamma_prime
from .constant_asymptotics import LaplaceConstant
from src.risk_computations import RiskComputations
from src.least_squares import PowerLawRegression
from src.SGD import SGD
import numpy as np
from scheduled import WSDSchedule
from scipy.special import gamma, gammainc, gammaincc



class LaplaceLinear(AsymptoticsAnalysis):
    def __init__(self, model: PowerLawRegression, x0, T_max=100000, optimize=True, base_lr=0.01):
        super().__init__(model, x0)
        print(f"Initializing Laplace_linear with T_max={T_max} for setup...")
        self._setup_for_T(T_max, optimize=optimize, base_lr=base_lr) 

    

    def _setup_for_T(self, T, optimize=True, base_lr=0.01):
        """Configure the schedule and computations for a specific horizon T."""
        self.schedule = WSDSchedule(steps=T, base_lr=base_lr, cooldown_len=1.)
        self.sgd = SGD(self.model, self.x0, self.schedule)
        self.computations = RiskComputations(
            self.model, self.x0, [self.schedule], ["linear"], sgd_class=SGD
        )
        
        # Optimize learning rate specifically for this T
        if optimize:
            self.computations.optimize_all_base_lrs(t_value=T-1, change_eta=True)

    def _update_schedule_for_T(self, T):
        """Update the schedule and re-optimize eta for a new T."""
        assert self.schedule is not None, "Schedule must be initialized before updating."
        self.schedule = WSDSchedule(steps=T, base_lr=self.schedule.get_base_lr(), cooldown_len=1.)
        self.sgd.schedule = self.schedule

    def compute_laplace_approx_bias(self, T, t, m_exponent, m_constant):
        """Compute the bias term for the linear schedule using the Laplace approximation at step t."""
        eta = self.schedule.get_base_lr() 
        alpha = self.model.exponent
        
        # Calculate 1st-order bias component
        C1 = (m_exponent - 1) / alpha + 1
        
        # Calculate bias_base using the generalized formula for step t
        # Assuming L = 1 as in the original code structure
        bias_base = 2 * eta * (t - (t * (t - 1)) / (2 * T))
        
        # Handle edge case at initialization to avoid division by zero
        if bias_base <= 0:
            return float('inf')
            
        bias = (m_constant / (2 * alpha)) * gamma(C1) / (bias_base ** C1)

        return bias
    
    def _compute_laplace_approx_variance_legacy(self, T, t):
        """Legacy discrete-time variance approximation kept for reference."""
        eta = self.schedule.get_base_lr() 
        sigma_sq = self.model.sigma**2
        alpha = self.model.exponent
        
        # Setup constants for variance calculation
        Cv1 = (2 * alpha - 1) / alpha
        variance_prefix = (sigma_sq * eta**2) / (2 * alpha)
        
        # Variance is 0 before any steps are taken
        if t <= 0:
            return 0.0
        
        k = np.arange(t - 1)
        
        var_base = (2 * eta / T) * (t - k - 1) * (T - (t + k) / 2)
        
        terms_1st = ((T - k) / T)**2 / (var_base ** Cv1)
        sum_terms_1st = np.sum(terms_1st) * gamma(Cv1)
        
        # Special case for the last term (k = t - 1) where var_base becomes 0
        #last_term = ((T - (t - 1)) / T)**2 * (alpha / (2 * alpha - 1))
        last_term = ((T - (t - 1)) / T)**2 * (alpha / (2 * alpha - 1))

        variance = variance_prefix * (sum_terms_1st + last_term)

        return variance

    def compute_laplace_approx_variance(self, T, t, *args, **kwargs):
        """Main variance method: use the double-integral formula at final time only."""
        t_int = int(t)
        assert t_int >= T - 1, (
            "compute_laplace_approx_variance now uses the double-integral formula, "
            "implemented for final time only (t >= T-1)."
        )
        #return self.compute_laplace_approx_variance_double_integral(T, K=1)
        return self.compute_laplace_approx_variance_dim(T)

    def compute_laplace_approx_risk_for_T(self, T, t, m_exponent, m_constant):
        """Setup for T, optimize eta, and compute 1st-order Laplace approximate risk for a linear schedule."""
        bias = self.compute_laplace_approx_bias(T, t, m_exponent, m_constant)
        variance = self.compute_laplace_approx_variance(T, t, m_exponent, m_constant)       
        return bias + variance
    

    def compute_laplace_approx_biases_and_variances_different_finals(self, T_values, m_exponent, m_constant, K=1):
        """Compute both bias and variance components using the double-integral variance (K=1 only)."""
        assert 0<= K <= 1, "K should be between 0 and 1 to ensure t=K*T is a valid step within the schedule."
        assert K == 1, "This method now uses the double-integral variance only, which is implemented for K=1."
        
        biases, variances = {}, {}
        for T in T_values:
            self.schedule.steps = T  # Update schedule for new T
            #self.computations.optimize_all_base_lrs(t_value=T-1, change_eta=True)  # Re-optimize eta for new T
            t = int(K * (T-1))
            bias = self.compute_laplace_approx_bias(T, t, m_exponent, m_constant)
            variance = self.compute_laplace_approx_variance_double_integral(T, K)
            biases[T] = bias
            variances[T] = variance
        return biases, variances


    def compute_laplace_approx_variance_partial(self, T, t, *args, **kwargs):
        """
        Compute the variance term for the linear schedule using the exact 
        integral bounds via the incomplete gamma function.
        """
        eta = self.schedule.get_base_lr()
        sigma_sq = self.model.sigma**2
        alpha = self.model.exponent
        
        # Setup constants for variance calculation
        Cv1 = (2 * alpha - 1) / alpha
        L = 1.0 
        variance_prefix = (L**2 * sigma_sq * eta**2) / (2 * alpha)
        
        # Variance is 0 before any steps are taken
        if t <= 0:
            return 0.0
        
        k = np.arange(t - 1)
        
        # Calculate the exponent base C
        var_base = (2 * eta * L / T) * (t - k - 1) * (T - (t + k) / 2)
        
        terms_1st = ((T - k) / T)**2 / (var_base ** Cv1)
        
        # Correction: Multiply by gammainc to truncate the integral exactly at z=1
        exact_integrals = terms_1st * gamma(Cv1) * gammainc(Cv1, var_base)
        sum_terms_1st = np.sum(exact_integrals)
        
        # Special case for the last term (k = t - 1)
        last_term = ((T - (t - 1)) / T)**2 * (alpha / (2 * alpha - 1))
        
        variance = variance_prefix * (sum_terms_1st + last_term)
        
        return variance


    def compute_laplace_approx_variance_double_integral(self, T, K,  *args, **kwargs):
        """Compute the variance term for the linear schedule using a double integral approach."""
        assert K==1, "Double integral variance computation currently only implemented for K=1 (t=T)."
        
        eta = self.schedule.get_base_lr()
        alpha = self.model.exponent
        L=1.
        
        if alpha != 2:

            prefix1 = (L**2 * self.model.sigma**2 * eta**2) / (2 * alpha)
            prefix2 = T*alpha/(alpha-2)

            term1 = gamma(1.5)/(eta*L*T)**1.5
            term2 = - gamma((2*alpha-1)/alpha) / (eta*L*T)**((2*alpha-1)/alpha)

            variance = prefix1 * prefix2 * (term1 + term2)

        else:

            prefix1 = (L**2 * self.model.sigma**2 * eta**2) / (2 * alpha)
            prefix2 = T / (2*(eta*L*T)**1.5)

            term1 = gamma(1.5)*np.log(eta*L*T)
            term2 = - gamma_prime(1.5)

            variance = prefix1 * prefix2 * (term1 + term2)
        return variance
    

    def compute_laplace_approx_variance_dim(self, T):
        """
        Compute the asymptotic two-term expansion for the variance based on:
        V_t * (L^2 * sigma^2 * eta^2) / (2 * alpha) approx ...
        """
        eta = self.schedule.get_base_lr()
        alpha = self.model.exponent
        L = 1.0
        sigma = self.model.sigma
        dim = self.model.dim
        tau = T/dim**alpha


        if alpha == 2:
            prefix1 = (L**2 * self.model.sigma**2 * eta**2) / (2 * alpha)
            prefix2 = T / (2*(eta*L*T)**1.5)

            term1 = gamma(1.5)*np.log(eta*L*T)
            term2 = - gamma_prime(1.5)

            variance = prefix1 * prefix2 * (term1 + term2)
        else:
                
            # Pre-factor (alpha / (alpha - 2))
            pre_factor = alpha / (alpha - 2.0)

            gamma_term = gamma(1.5) * gammainc(1.5, eta*L*tau) / (eta*L*tau)**1.5

            # First term: (tau^(3/2) / T^(1/2)) * pre_factor * integral
            term_1 = (tau**(1.5) / np.sqrt(T)) * pre_factor * gamma_term

            # Second term components
            # Part A: Gamma(3/2) / (eta * L)^(3/2)
            part_a = gamma(1.5) / ((eta * L)**1.5)
            
            # Part B: (1/T)^((alpha - 2) / 2alpha) * Gamma((alpha - 2)/2alpha + 3/2) / (eta * L)^((alpha - 2)/2alpha + 3/2)
            exponent_b = (alpha - 2.0) / (2.0 * alpha)
            power_b = exponent_b + 1.5
            part_b = (1.0 / T)**exponent_b * gamma(power_b) / ((eta * L)**power_b)

            # Combined second term
            term_2 = (1.0 / np.sqrt(T)) * pre_factor * (part_a - part_b)

        
            # Result assembly: V_t = ((L**2 * sigma**2 * eta**2) / (2.0 * alpha))  * (term_1 + term_2)
            variance = ((L**2 * sigma**2 * eta**2) / (2.0 * alpha)) * (term_1 + term_2)
        
        return variance




#End of class definitions


def compute_different_sigmas(T, model, x0, Delta, beta, sigmas, schedule_type="constant"):
    """Compute Laplace risk approximation for different noise levels."""
    results = {}
    real_approx = {}
    
    for sigma in sigmas:
        print(f"Computing for sigma={sigma}...")
        model.sigma = sigma  # Update the noise level in the model
        analysis = LaplaceConstant(model, x0, T) if schedule_type == "constant" else LaplaceLinear(model, x0, T)
        risk = analysis.compute_laplace_for_several_ts(T, Delta, beta)
        real_approx[sigma] = analysis.compute_real_approx_for_several_ts(T)
        results[sigma] = risk
        print(f"Laplace risk approximation for sigma={sigma} computed.")
        
    return results, real_approx
