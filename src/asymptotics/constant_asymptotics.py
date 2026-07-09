from .base_asymptotics import AsymptoticsAnalysis
from src.risk_computations import RiskComputations
from src.least_squares import PowerLawRegression
from src.SGD import SGD
import numpy as np
from scheduled import ConstantSchedule
from scipy.special import gamma, zeta
from scipy.optimize import minimize_scalar


class LaplaceConstant(AsymptoticsAnalysis):
    def __init__(self, model: PowerLawRegression, x0, T_max=100000, optimize=False, base_lr=0.01, beta=None):
        # Initialize without a fixed T
        if beta is None:
            beta = model.exponent
        super().__init__(model, x0, beta=beta)
        print(f"Initializing Laplace_constant with T_max={T_max} for setup...")
        self._setup_for_T(T_max, optimize=optimize, base_lr=base_lr)  # Setup for a large T to compute m0 and optimize eta


    def _setup_for_T(self, T, optimize=False, base_lr=0.01):
        """Configure the schedule and computations for a specific horizon T."""
        self.schedule = ConstantSchedule(steps=T, base_lr=base_lr)
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
        self.schedule = ConstantSchedule(steps=T, base_lr=self.schedule.get_base_lr())
        self.sgd.schedule = self.schedule

    def compute_laplace_approx_risk_for_T(self, T, m_exponent, m_constant, separate_bias_variance=False):
        """Setup for T, optimize eta, and compute Lagrange approximate risk."""
        self._update_schedule_for_T(T)
        t = T - 1  # Evaluate at the last step of the schedule
        bias = self.compute_laplace_approx_bias(T, t, m_exponent, m_constant)
        variance = self.compute_laplace_approx_variance(T, t)
        if separate_bias_variance:
            return bias, variance
        return bias + variance

    def compute_laplace_approx_variance(self, T, t):
        """Compute the variance term for the constant schedule using the Laplace approximation at step t."""
        eta = self.schedule.get_base_lr() 
        sigma_sq = self.model.sigma**2
        alpha = self.model.exponent
        
        variance = eta * sigma_sq / (4 * (alpha - 1))
        
        return variance
    
    def compute_laplace_approx_bias(self, T, t, m_exponent, m_constant):
        """Compute the bias term for the constant schedule using the Laplace approximation at step t."""
        eta = self.schedule.get_base_lr() 
        alpha = self.model.exponent

        C = (m_exponent - 1) / alpha + 1
        bias = m_constant / (2 * alpha) * gamma(C) / (2 * eta * T)**C
        
        return bias
    

    

class SlockConstant(LaplaceConstant):

    def __init__(self, model: PowerLawRegression, x0, T_max=100000, optimize=False, base_lr=0.01, beta=None):
        # Reuse parent initialization and preserve optional beta override.
        super().__init__(model, x0, T_max=T_max, optimize=optimize, base_lr=base_lr, beta=beta)


    def _setup_for_T(self, T, optimize=False, base_lr=0.01):
        """Configure the schedule and computations for a specific horizon T."""
        self.schedule = ConstantSchedule(steps=T, base_lr=base_lr)
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
        self.schedule = ConstantSchedule(steps=T, base_lr=self.schedule.get_base_lr())
        self.sgd.schedule = self.schedule



    def compute_slock_approx_bias(self, T, t, m_exponent, m_constant, batch=1):
        """Compute the bias term for the constant schedule using the SLOCK approximation at step t."""
        eta = self.schedule.get_base_lr() 
        alpha = self.model.exponent

        eta_bar = eta - 0.5*eta**2 * np.sum(self.model.Lambda_vals) / batch 

        C = (m_exponent - 1) / alpha + 1
        bias = m_constant / (2 * alpha) * gamma(C) / (2 * eta_bar * T)**C
        
        return bias

    def compute_slock_approx_variance(self, T, t, batch=1):
        """Compute the variance term for the constant schedule using the SLOCK approximation at step t."""
        eta = self.schedule.get_base_lr() 
        sigma_sq = self.model.sigma**2
        alpha = self.model.exponent
        L = 1.

        eta_bar = eta - 0.5*eta**2 * np.sum(self.model.Lambda_vals) / batch

        prefix1 = (eta**2 * sigma_sq*L)/(4*eta_bar )

        variance = prefix1 * zeta(alpha) / batch

        return variance
    

    def compute_laplace_approx_risk_for_T(self, T,t, m_exponent, m_constant, separate_bias_variance=False):
        """Setup for T, optimize eta, and compute Lagrange approximate risk."""
        self._update_schedule_for_T(T)
        t = T - 1  # Evaluate at the last step of the schedule
        bias = self.compute_slock_approx_bias(T, t, m_exponent, m_constant)
        variance = self.compute_slock_approx_variance(T, t)
        if separate_bias_variance:
            return bias, variance
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
        Compute the optimal learning rate (eta^*) for the Constant schedule
        using the exact closed-form SLOCK approximations.
        """
        beta = self.beta
        Delta = m_constant
        alpha = self.model.exponent
        sigma_sq = self.model.sigma**2
        L = 1.0


        # 1. Compute the recurring bias exponent omega
        omega = (alpha + beta - 1.0) / alpha

        # 2. Compute the constant factor C_cst
        c_cst_numerator = omega * gamma(omega)
        c_cst_denominator = alpha * zeta(alpha) * (2.0 ** (omega - 1.0))
        c_cst = c_cst_numerator / c_cst_denominator

        # 3. Compute optimal eta components
        numerator = c_cst * Delta
        denominator = sigma_sq * (L ** omega)
        
        # Calculate the prefix exponent and T exponent
        exponent_denominator = 2.0 * alpha + beta - 1.0
        prefix_pow = alpha / exponent_denominator
        t_exponent = -(alpha + beta - 1.0) / exponent_denominator

        # 4. Final calculation
        prefix = (numerator / denominator) ** prefix_pow
        eta_star = prefix * (T ** t_exponent)
        
        return eta_star
    
    def compute_exact_eta(self, T, m_constant, batch=1):
        """Without a closed form, without assuming (1-eta*tr...) ~ 1"""
        beta = self.beta
        Delta = m_constant
        alpha = self.model.exponent
        sigma_sq = self.model.sigma**2
        L = 1.0
        omega = (beta - 1.0) / alpha
        tr = np.sum(self.model.Lambda_vals) 

        c_bias = L * Delta * gamma(omega+1) / (2*alpha * (2*L*T)**(omega+1))
        c_var = (sigma_sq * zeta(alpha) * L) / (4*batch)

        def risk(eta):
            """Compute the total risk for a given learning rate eta. (T, Delta... fixed)"""
            one_minus_eta = 1 - eta * tr / (2 * batch)
            return c_bias / ((eta * one_minus_eta) ** (omega + 1)) + c_var * eta / one_minus_eta
        
        # Define the strict bounds to prevent division by zero or negative bases
        epsilon = 1e-12
        upper_bound = (2.0 * batch) / tr - epsilon
        
        # Perform bounded numerical optimization
        result = minimize_scalar(
            risk, 
            bounds=(epsilon, upper_bound), 
            method='bounded',
            options={'xatol': 1e-10} # High precision tolerance
        )
        
        # Check if the optimization was successful
        if result.success:
            optimal_eta = result.x
            minimum_risk = result.fun
            return optimal_eta
        else:
            raise ValueError("Optimization failed to converge.")