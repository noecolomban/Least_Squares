from .base_asymptotics import AsymptoticsAnalysis, gamma_prime
from src.risk_computations import RiskComputations
from src.least_squares import PowerLawRegression
from src.SGD import SGD
import numpy as np
from scheduled import WSDSchedule
from scipy.special import gamma, zeta
from src.utils import constant_zeta_correction

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
            self.model, self.x0, [self.schedule], ["wsd"], sgd_class=SGD
        )
        
        # Optimize learning rate specifically for this T
        if optimize:
            self.computations.optimize_all_base_lrs(t_value=T-1, change_eta=True)


    def _update_schedule_for_T(self, T, new_eta=None):
        """Update the schedule and re-optimize eta for a new T."""
        assert self.schedule is not None, "Schedule must be initialized before updating."
        if new_eta is None:
            new_eta = self.schedule.get_base_lr()
        self.schedule = WSDSchedule(steps=T, base_lr=new_eta, cooldown_len=self.cooldown_len)
        self.sgd.schedule = self.schedule
        self.computations = RiskComputations(
            self.model, self.x0, [self.schedule], ["wsd"], sgd_class=SGD
        )

    def compute_laplace_approx_risk_for_T(self, T, t, m_exponent, m_constant):
        """Setup for T, optimize eta, and compute 1st-order Laplace approximate risk for a linear schedule."""
        bias = self.compute_laplace_approx_bias(T, t, m_exponent, m_constant)
        K = t / T
        variance = self.compute_laplace_approx_variance(T, K, m_exponent, m_constant)       
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
    

    def compute_laplace_approx_variance(self, T, t, *args, corrected=True, **kwargs):
        """
        Computes the Laplace approximation variance (V_t).
        """
        K=1
        eta = self.schedule.get_base_lr()
        sigma_sq = self.model.sigma**2
        alpha = self.model.exponent
        
        L = 1.0         
        KT = int(K * (T-1))
        T0 = self.T0(T)  # cooldown start step
        
        # A_t (Variance accumulated before the decay phase) 
        # Calculate the exponent used in multiple places: (alpha - 1) / alpha
        exponent = (alpha - 1) / alpha
        gamma_val = gamma(exponent)
        
        # Fraction: (KT - T0 - 1) * (2T - T0 - KT) / (T - T0)
        fraction_num = (KT - T0 - 1) * (2 * T - T0 - KT)
        fraction_den = max(1e-9, T - T0)  # Prevent division by zero
        common_fraction = fraction_num / fraction_den
        
        inner_base_1 = eta * L * common_fraction
        inner_base_2 = 2 * eta * L * (T0 + 1) + inner_base_1
        
        # Evaluate the terms inside the square brackets
        term_1 = gamma_val / (inner_base_1 ** exponent) if inner_base_1 > 0 else 0
        term_2 = gamma_val / (inner_base_2 ** exponent)
        bracket_result = term_1 - term_2
        
        prefix = (L * eta * sigma_sq) / (4 * alpha)
        A_t = prefix * bracket_result
        
        # B_t (Variance accumulated during the decay phase)
       
        # Exact double integral correction for K=1 boundary
        T_decay = T - T0
        s = eta * L * T_decay
        G = (L**2) * sigma_sq * (eta**2) * T_decay
        
        # Use a small tolerance for floating point comparison with 2.0
        if abs(alpha - 2.0) < 1e-9:
            term_1_b = gamma_prime(1.5)
            term_2_b = np.log(s) * gamma(1.5)
            B_t = - (G / (8 * (s ** 1.5))) * (term_1_b - term_2_b)
        else:
            term_1_b = gamma(1.5) / (s ** 1.5)
            exp_alpha = (2 * alpha - 1) / alpha
            term_2_b = gamma(exp_alpha) / (s ** exp_alpha)
            B_t = (G / (2 * (alpha - 2))) * (term_1_b - term_2_b)
        
        if not corrected:
            variance = A_t + B_t
        else:
            B_t_corrected = constant_zeta_correction(alpha)*B_t
            #A_t_corrected = constant_zeta_correction(2*alpha)*A_t
            A_t_corrected = A_t  # No correction for A_t, only for B_t, as the double integral correction is more relevant for the decay phase variance
            variance = A_t_corrected + B_t_corrected        
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