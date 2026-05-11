from sys import prefix

from matplotlib.pylab import seed

from src.risk_computations import RiskComputations
from src.least_squares import PowerLawRegression
from src.SGD import SGD
import numpy as np
from scheduled import WSDSchedule, ConstantSchedule
from scipy.special import gamma, gammainc, digamma
from abc import ABC, abstractmethod


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
        self.computations = None

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


class LaplaceConstant(AsymptoticsAnalysis):
    def __init__(self, model: PowerLawRegression, x0, T_max=100000):
        # Initialize without a fixed T
        super().__init__(model, x0)
        print(f"Initializing Laplace_constant with T_max={T_max} for setup...")
        self._setup_for_T(T_max)  # Setup for a large T to compute m0 and optimize eta


    def _setup_for_T(self, T):
        """Configure the schedule and computations for a specific horizon T."""
        self.schedule = ConstantSchedule(steps=T, base_lr=0.1)
        self.sgd = SGD(self.model, self.x0, self.schedule)
        self.computations = RiskComputations(
            self.model, self.x0, [self.schedule], ["constant"], sgd_class=SGD
        )
        
        # Optimize learning rate specifically for this T
        self.computations.optimize_all_base_lrs(t_value=T-1, change_eta=True)


    def _update_schedule_for_T(self, T):
        """Update the schedule for a new T."""
        assert self.schedule is not None, "Schedule must be initialized before updating."
        self.schedule = ConstantSchedule(steps=T, base_lr=self.schedule.get_base_lr())
        self.sgd.schedule = self.schedule

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
    
    def compute_laplace_approx_variance(self, T, t, *args, **kwargs):
        """Compute the variance term for the linear schedule using the Laplace approximation at step t."""
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
    

    def compute_laplace_approx_risk_for_T(self, T, t, m_exponent, m_constant):
        """Setup for T, optimize eta, and compute 1st-order Laplace approximate risk for a linear schedule."""
        bias = self.compute_laplace_approx_bias(T, t, m_exponent, m_constant)
        variance = self.compute_laplace_approx_variance(T, t, m_exponent, m_constant)       
        return bias + variance
    

    def compute_laplace_approx_biases_and_variances_different_finals(self, T_values, m_exponent, m_constant, K=1):
        """Compute both bias and variance components separately for analysis. For different t=K*T"""
        assert 0<= K <= 1, "K should be between 0 and 1 to ensure t=K*T is a valid step within the schedule."
        
        biases, variances = {}, {}
        for T in T_values:
            self.schedule.steps = T  # Update schedule for new T
            #self.computations.optimize_all_base_lrs(t_value=T-1, change_eta=True)  # Re-optimize eta for new T
            t = int(K * (T-1))
            bias = self.compute_laplace_approx_bias(T, t, m_exponent, m_constant)
            #variance = self.compute_laplace_approx_variance(T, t, m_exponent, m_constant)       
            #variance = self.compute_laplace_approx_variance_partial(T, t)
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


class LaplaceWSD(AsymptoticsAnalysis):
    """Class for analyzing the Laplace approximation specifically for the WSD schedule."""
    def __init__(self, model: PowerLawRegression, x0, T_max=100000, optimize=True, base_lr=0.01, cooldown_len=0.2):
        super().__init__(model, x0)
        print(f"Initializing LaplaceWSD with T_max={T_max} for setup...")
        self._setup_for_T(T_max, optimize=optimize, base_lr=base_lr, cooldown_len=cooldown_len)
        self.cooldown_len = cooldown_len

    def T0(self, T):
        """Compute the cooldown start step T0 based on the total steps T."""
        return int((1-self.cooldown_len) * T)

    def _setup_for_T(self, T, optimize=True, base_lr=0.01, cooldown_len=0.2):
        """Configure the schedule and computations for a specific horizon T."""
        self.schedule = WSDSchedule(steps=T, base_lr=base_lr, cooldown_len=cooldown_len)
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
        self.schedule = WSDSchedule(steps=T, base_lr=self.schedule.get_base_lr(), cooldown_len=self.cooldown_len)
        self.sgd.schedule = self.schedule

    def compute_laplace_approx_risk_for_T(self, T, t, m_exponent, m_constant):
        """Setup for T, optimize eta, and compute 1st-order Laplace approximate risk for a linear schedule."""
        bias = self.compute_laplace_approx_bias(T, t, m_exponent, m_constant)
        variance = self.compute_laplace_approx_variance(T, t, m_exponent, m_constant)       
        return bias + variance
    

    def compute_laplace_approx_bias(self, T, t, m_exponent, m_constant):
        """
        Computes the Laplace approximation bias.
        """

        L=1.
        T0 = self.T0(T)  # cooldown start step
        alpha = self.model.exponent
        eta = self.schedule.get_base_lr()

        # m_exponent represents beta
        exponent = ((m_exponent - 1) / alpha) + 1
        
        # m_constant represents Delta
        actual_constant_multiplier = (L * m_constant) / (2 * alpha)

        time_numerator = (t - T0) * (2 * T - T0 - t + 1)
        time_denominator = 2 * (T - T0)
        time_fraction = time_numerator / time_denominator
        
        inner_base = 2 * eta * L * (T0 + time_fraction)
        
        denominator = inner_base ** exponent
        
        numerator = actual_constant_multiplier * gamma(exponent)
        
        # Final computation
        bias = numerator / denominator
        
        return bias
    

    def compute_laplace_approx_variance(self, T, K, *args, **kwargs):
        """
        Computes the Laplace approximation variance (V_t).
        Assumes 'KT' in the mathematical formula corresponds to the variable 't'.
        """
        eta = self.schedule.get_base_lr()
        sigma_sq = self.model.sigma**2
        alpha = self.model.exponent
        
        L = 1.0         
        KT = int(K * (T-1))
        
        # Calculate the exponent used in multiple places: (alpha - 1) / alpha
        exponent = (alpha - 1) / alpha
        gamma_val = gamma(exponent)
        
        T0 = self.T0(T)  # cooldown start step
        
        # Fraction: (KT - T0 - 1) * (2T - T0 - KT) / (T - T0)
        fraction_num = (KT - T0 - 1) * (2 * T - T0 - KT)
        fraction_den = T - T0
        common_fraction = fraction_num / fraction_den
        
        inner_base_1 = eta * L * common_fraction
        inner_base_2 = 2 * eta * L * (T0 + 1) + inner_base_1
        
        # Evaluate the terms inside the square brackets
        term_1 = gamma_val / (inner_base_1 ** exponent)
        term_2 = gamma_val / (inner_base_2 ** exponent)
        bracket_result = term_1 - term_2
        

        prefix = (L * eta * sigma_sq) / (4 * alpha)
        
        # Additive: (L * eta * sigma^2 * (T - KT)) / (4 * (alpha - 1) * (T - T0))
        additive_num = L * eta * sigma_sq * (T - KT)
        additive_den = 4 * (alpha - 1) * (T - T0)
        additive_term = additive_num / additive_den
        
        variance = (prefix * bracket_result) + additive_term        
        return variance
    
    def compute_laplace_approx_biases_and_variances_different_finals(self, T_values, m_exponent, m_constant, K=1):
        """Compute both bias and variance components separately for analysis. For different t=K*T"""
        assert 0<= K <= 1, "K should be between 0 and 1 to ensure t=K*T is a valid step within the schedule."
        assert all(K*T > self.T0(T) for T in T_values), "K*T must be greater than the cooldown start step T0 for the bias and variance formulas to be valid."

        biases, variances = {}, {}
        for T in T_values:
            self.schedule.steps = T  # Update schedule for new T
            t = int(K * (T-1))
            bias = self.compute_laplace_approx_bias(T, t, m_exponent, m_constant)
            variance = self.compute_laplace_approx_variance(T, K)
            biases[T] = bias
            variances[T] = variance
        return biases, variances