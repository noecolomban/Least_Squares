from .base_asymptotics import AsymptoticsAnalysis
from src.risk_computations import RiskComputations
from src.least_squares import PowerLawRegression
from src.SGD import SGD
import numpy as np
from scheduled import ConstantSchedule
from scipy.special import gamma


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
