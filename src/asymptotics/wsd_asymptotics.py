from .base_asymptotics import AsymptoticsAnalysis, gamma_prime
from src.risk_computations import RiskComputations
from src.least_squares import PowerLawRegression
from src.SGD import SGD
import numpy as np
from scheduled import WSDSchedule
from scipy.special import gamma, zeta
from scipy.integrate import quad
from src.utils import constant_zeta_correction

class LaplaceWSD(AsymptoticsAnalysis):
    """Class for analyzing the Laplace approximation specifically for the WSD schedule."""
    def __init__(self, model: PowerLawRegression, x0, T_max=100000, optimize=False, base_lr=0.01, cooldown_len=0.2):
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
    
    def set_eta_star(self, T, m_constant) -> float:
        """Compute and set the optimal learning rate (eta^*) for the WSD schedule."""
        eta_star = self.compute_best_slock_eta(T, m_constant)
        self._update_schedule_for_T(T, new_eta=eta_star)
        return eta_star

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
        beta = m_exponent
        Delta = m_constant

        # m_exponent represents beta
        exponent = ((beta - 1) / alpha) + 1
        
        # m_constant represents Delta
        actual_constant_multiplier = (L * Delta) / (2 * alpha)

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
    


class SlockWSD(AsymptoticsAnalysis):

    def __init__(self, model: PowerLawRegression, x0, beta, T_max=100000, base_lr=0.01, cooldown_len=0.2, optimize=False):
        # Initialize without a fixed T
        super().__init__(model, x0, beta)
        print(f"Initializing Laplace_constant with T_max={T_max} for setup...")
        self._setup_for_T(T_max, cooldown_len=cooldown_len, optimize=optimize, base_lr=base_lr)  # Setup for a large T to compute m0 and optimize eta
        self.cooldown_len = cooldown_len  

    def _setup_for_T(self, T, optimize=False, base_lr=0.01, cooldown_len=None):
        """Configure the schedule and computations for a specific horizon T."""
        if cooldown_len is None:
            cooldown_len = self.cooldown_len
        self.schedule = WSDSchedule(steps=T, base_lr=base_lr, cooldown_len=cooldown_len)
        self.sgd = SGD(self.model, self.x0, self.schedule)
        self.computations = RiskComputations(
            self.model, self.x0, [self.schedule], ["constant"], sgd_class=SGD
        )
        
        # Optimize learning rate specifically for this T
        if optimize:
            self.computations.optimize_all_base_lrs(t_value=T-1, change_eta=True)


    def _update_schedule_for_T(self, T):
        """Update the schedule for a new T."""
        assert self.schedule is not None, "Schedule must be initialized before updating."
        self.schedule = WSDSchedule(steps=T, base_lr=self.schedule.get_base_lr(), cooldown_len=self.cooldown_len)
        self.sgd.schedule = self.schedule




    def compute_slock_approx_bias(self, T, t, m_exponent, m_constant):
        """Compute the bias term for the WSD schedule using the SLOCK approximation."""
        eta = self.schedule.get_base_lr() 
        alpha = self.model.exponent
        cl = self.cooldown_len

        # Calculate the trace of the Lambda matrix
        tr_lambda = np.sum(self.model.Lambda_vals)

        # WSD specific factor for the denominator base
        wsd_factor = 2 - cl - eta * tr_lambda * (1 - 2 * cl / 3)
        denominator_base = eta * T * wsd_factor

        # Asymptotic power C = (beta - 1) / alpha + 1
        C = (m_exponent - 1) / alpha + 1
        
        # Final bias calculation
        bias = (m_constant / (2 * alpha)) * gamma(C) / (denominator_base)**C
        
        return bias



    def compute_slock_approx_variance(self, T, t):
        """
        Compute the variance term for the WSD schedule using the SLOCK approximation.
        Handles the phase transition at alpha = 2.
        """
        eta = self.schedule.get_base_lr() 
        sigma_sq = self.model.sigma**2
        alpha = self.model.exponent
        L = 1.0  # Assuming L=1 as in your original function
        cl = self.cooldown_len

        tr_lambda = np.sum(self.model.Lambda_vals)

        if alpha > 2.0:
            # Asymptotic regime for alpha > 2: dominated purely by the cooldown phase (O(T^-1/2))
            prefix = (np.sqrt(np.pi) / 8.0) * np.sqrt(L * eta) * sigma_sq
            variance = prefix * zeta(alpha / 2.0) * (cl * T)**(-0.5)
            
            return variance

        elif 1.0 < alpha < 2.0:
            # Asymptotic regime for 1 < alpha < 2: both constant and cooldown phases contribute (O(T^((1-alpha)/alpha)))
            A = 2 * eta * L
            B = tr_lambda * (eta**2) * L
            D = B / 3.0 - A / 2.0
            
            kappa_1 = (A - B) * (1 - cl) - cl * D
            kappa_2 = -cl * D

            # Phase A contribution (Constant phase)
            term_A_prefix = (L**2 * eta**2 * sigma_sq) / (2 * alpha * (B - A))
            term_A_gamma = gamma((alpha - 1) / alpha)
            term_A_brackets = kappa_1**((1 - alpha) / alpha) - kappa_2**((1 - alpha) / alpha)
            var_A = term_A_prefix * term_A_gamma * term_A_brackets
            
            # Phase B contribution (Cooldown phase)
            def integrand(x):
                return (x**((2 - 2*alpha) / alpha)) * ((1 - (eta * tr_lambda / 3.0) * x)**((1 - 2*alpha) / alpha))
            
            I_alpha_eta, _ = quad(integrand, 0, 1)
            
            term_B_prefix = (L**2 * eta**2 * sigma_sq) / (2 * alpha)
            term_B_gamma = gamma(2 - 1.0 / alpha)
            term_B_power = (eta * L)**((1 - 2*alpha) / alpha)
            var_B = term_B_prefix * term_B_gamma * term_B_power * I_alpha_eta * (cl**((1 - alpha) / alpha))
            
            # Total variance combined with the temporal scaling
            variance = (var_A + var_B) * (T**((1 - alpha) / alpha))
            
            return variance

        else:
            # Handle edge cases not covered by the asymptotic bounds (alpha <= 1 or alpha == 2)
            raise ValueError("Alpha must be strictly in (1, 2) or strictly > 2 for this SLOCK approximation.")
    

    def compute_laplace_approx_risk_for_T(self, T,t, m_exponent, m_constant, separate_bias_variance=False):
        """Setup for T, optimize eta, and compute Lagrange approximate risk."""
        self._update_schedule_for_T(T)
        t = T - 1  # Evaluate at the last step of the schedule
        bias = self.compute_slock_approx_bias(T, t, m_exponent, m_constant)
        variance = self.compute_slock_approx_variance(T, t)
        if separate_bias_variance:
            return bias, variance
        return bias + variance
    
    def compute_slock_approx_risk(self, T, m_constant, m_exponent=None):
        """Compute the SLOCK approximate risk for the WSD schedule at the last step.""" 
        if m_exponent is None:
            m_exponent = self.beta
        self._update_schedule_for_T(T)
        bias = self.compute_slock_approx_bias(T, T - 1, m_exponent, m_constant)
        variance = self.compute_slock_approx_variance(T, T - 1)
        return bias + variance


    def compute_laplace_approx_bias(self, T, t, m_exponent, m_constant):
        """Compute the bias term for the constant schedule using the Laplace approximation at step t."""
        bias, variance = self.compute_laplace_approx_risk_for_T(T, t, m_exponent, m_constant, separate_bias_variance=True)
        return bias
    
    def compute_laplace_approx_variance(self, T, t):
        """Compute the variance term for the constant schedule using the Laplace approximation at step t."""
        bias, variance = self.compute_laplace_approx_risk_for_T(T, t, m_exponent=0, m_constant=0, separate_bias_variance=True)
        return variance


    def compute_best_slock_eta(self, T, m_constant):
        """
        Compute the optimal learning rate (eta^*) for the WSD schedule
        using the exact closed-form SLOCK approximations.
        """
        beta = self.beta
        Delta = m_constant
        alpha = self.model.exponent
        sigma_sq = self.model.sigma**2
        L = 1.0
        cl = self.cooldown_len

        # Recurring bias exponent omega
        omega = (alpha + beta - 1.0) / alpha

        if 1.0 < alpha < 2.0:
            # ---------------------------------------------------------
            # Regime 1 < alpha < 2
            # ---------------------------------------------------------
            
            # 1. Compute the exact limit of the spatial integral I_{alpha, 0}
            I_alpha_0 = alpha / (2.0 - alpha)
            
            # 2. Compute the structural variance constant W(alpha, c)
            gamma_term = gamma(1.0 - 1.0 / alpha)
            bracket_term = I_alpha_0 * (cl ** ((1.0 - alpha) / alpha)) - ((2.0 - cl) ** ((1.0 - alpha) / alpha))
            W_alpha_c = 0.5 * gamma_term * bracket_term
            
            # 3. Compute optimal eta components
            numerator = omega * alpha * Delta * gamma(omega)
            denominator = sigma_sq * W_alpha_c * (L ** (beta / alpha)) * ((2.0 - cl) ** omega)
            
            prefix = (numerator / denominator) ** (alpha / (alpha + beta))
            exponent = -beta / (alpha + beta)
            
            eta_star = prefix * (T ** exponent)
            
        elif alpha > 2.0:
            # ---------------------------------------------------------
            # Regime alpha > 2
            # ---------------------------------------------------------
            
            numerator = 8.0 * omega * Delta * gamma(omega) * np.sqrt(cl)
            
            # L exponent: (alpha + 2*beta - 2) / (2*alpha)
            L_pow = (alpha + 2.0 * beta - 2.0) / (2.0 * alpha)
            denominator = alpha * np.sqrt(np.pi) * sigma_sq * zeta(alpha / 2.0) * (L ** L_pow) * ((2.0 - cl) ** omega)
            
            prefix_pow = (2.0 * alpha) / (3.0 * alpha + 2.0 * beta - 2.0)
            prefix = (numerator / denominator) ** prefix_pow
            
            exponent = (2.0 - alpha - 2.0 * beta) / (3.0 * alpha + 2.0 * beta - 2.0)
            
            eta_star = prefix * (T ** exponent)
            
        else:
            # Handle alpha <= 1 or alpha == 2
            raise ValueError("Alpha must be strictly in (1, 2) or strictly > 2 for this closed-form approximation.")

        return eta_star