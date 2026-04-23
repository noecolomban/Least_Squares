from unittest import result
from xml.dom.minidom import Element
from src.risk_computations import RiskComputations
from src.least_squares import PowerLawRegression
from src.SGD import SGD
import numpy as np
from scheduled import WSDSchedule, ConstantSchedule
from src.visualization import Visualization
import copy

class ZTransform_constant:
    def __init__(self, model: PowerLawRegression, x0, T=1000):
        self.model = model
        self.T = T
        self.x0 = x0
        self.m0 = self._compute_m0()
        self.schedule = ConstantSchedule(steps=T, base_lr=0.1)
        self.sgd = SGD(model, x0, self.schedule)
        self.computations = RiskComputations(model, x0, [self.schedule], ["constant"], sgd_class=SGD)

        self.computations.optimize_all_base_lrs(change_eta=True)

        self._z_transform_results = {}
        self._a = {}

    def _compute_m0(self):
        """Compute m0 = diag(Q^T * (x0 - x*) * (x0 - x*)^T * Q)"""
        Sigma0 = np.outer(self.x0.flatten() - self.model.x_star.flatten(), self.x0.flatten() - self.model.x_star.flatten())
        _, m0 = self.model.compute_M_t(Sigma0)
        return m0
    
    def a(self, i):
        assert 0 <= i < self.model.dim, "i must be between 0 and dim-1"
        if i in self._a:
            return self._a[i]
        else:
            eta = self.schedule.get_base_lr()
            Lambda_vals = self.model.Lambda_vals
            a = (1 - eta*Lambda_vals[i])**2 + 2*eta**2*Lambda_vals[i]**2
            self._a[i] = a
            return a


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

        z_results_values = {"constant" : self.compute_z_transform_result()*np.ones(self.T)}
        approx_risks = self.computations.approx_all_theoretical_risks()
        
        return z_results_values, approx_risks
        

def z_transform_several_ts(list_T = None, sigma=0.01, dim=100, eta_range=None):
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
        x0 = np.array([1/i**beta for i in range(1, dim+1)])

        schedules1 = [constant]
        asymptotics_analysis = ZTransform_constant(model, x0, T=T)

        # %%
        ztransform, approx = asymptotics_analysis.compute_all_approx_vs_z_transform()
        results[T] = (ztransform, approx)
    return results