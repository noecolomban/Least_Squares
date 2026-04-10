import numpy as np
import matplotlib.pyplot as plt
from scheduled.schedules.base import ScheduleBase
from src.least_squares import LinearRegression
from abc import ABC, abstractmethod


class BaseSGD(ABC):
    def __init__(self, model: LinearRegression, x0: np.ndarray, schedule: ScheduleBase):
        self.model = model
        self.x0 = x0.reshape(-1, 1)
        self._schedule = schedule
        self.T = schedule._steps
        self.L = self.model.Lambda_vals
        self.risks = {}
        self.losses = None
  

    def get_schedule(self):
        return self._schedule.schedule
    
    def get_step(self, t):
        assert self._schedule is not None, "Schedule is not defined"
        return self._schedule.schedule[t]

    def train(self, label="SGD", show=True):
        x = self.x0
        self.model.n_samples = self.T
        Phi, Y = self.model.generate_data()
        loss = []
        for t in range(self.T):
            phi, y = Phi[t].reshape(-1,1), Y[t]
            g = phi @ (np.dot(phi.T , x) - y)
            x = x - self.get_step(t) * g
            loss.append(self.model.compute_risk(x))
        self.losses = np.array(loss)
        if show:
            plt.plot(self.losses, label=label)
            plt.xlabel("Epoch")
            plt.ylabel("loss")
            plt.legend()
        return self.losses
    
    def compute_theoretical_risk(self, t) -> float:
        """
        Compute the risk at step T efficiently in O(T).
        """
        assert t < self.T, "T must be less than the number of training steps"

        if t not in self.risks:
            self.compute_all_theoretical_risks()
        return self.risks[t]

    @abstractmethod
    def compute_all_theoretical_risks(self) -> np.ndarray:
        pass



class SGD(BaseSGD):
    name = "True SGD"

    def __init__(self, model: LinearRegression, x0: np.ndarray, schedule: ScheduleBase):
        super().__init__(model, x0, schedule)
        self.list_At = {}

    def get_A_matrix(self, t):
        if t in self.list_At:
            return self.list_At[t]
        
        lr = self.get_step(t)
        term1 = np.diag((1 - lr * self.L)**2)
        term2 = np.diag(lr**2 * (self.L**2))
        term3 = (lr**2) * np.outer(self.L, self.L)
        self.list_At[t] = term1 + term2 + term3
        return self.list_At[t]


    def compute_all_theoretical_risks(self) -> np.ndarray:
        """
        Compute theoretical risks for all steps 0 to T in a single pass - O(T) instead of O(T²).
        Returns array of shape (T,) with risk at each step.
        """
        diff_0 = self.x0 - self.model.x_star
        Sigma_0 = diff_0 @ diff_0.T
        _, m_t = self.model.compute_M_t(Sigma_0)
        
        v_t = np.zeros(self.model.dim)
        risks = []
        irreducible_noise = self.model.sigma**2
        
        for t in range(self.T):
            bias_part = np.sum(self.L * m_t)
            variance_part = np.sum(self.L * v_t)
            risk = 0.5 * (bias_part + variance_part + irreducible_noise)
            self.risks[t] = risk
            risks.append(risk)
            
            # Update for next iteration
            lr = self.get_step(t)
            At = self.get_A_matrix(t)
            m_t = At @ m_t
            v_t = At @ v_t + (lr**2 * self.model.sigma**2) * self.L
        
        return np.array(risks)


class NoisyGD(BaseSGD):
    name = "Noisy GD"

    def __init__(self, model: LinearRegression, x0: np.ndarray, schedule: ScheduleBase):
        super().__init__(model, x0, schedule)
        self.list_At = {}

    def get_P_matrix(self, t):
        if t in self.list_At:
            return self.list_At[t]
        
        lr = self.get_step(t)
        term1 = np.diag((1 - lr * self.L)**2)
        self.list_At[t] = term1
        return self.list_At[t]

    def compute_all_theoretical_risks(self) -> np.ndarray:
        diff_0 = self.x0 - self.model.x_star
        Sigma_0 = diff_0 @ diff_0.T 
        _, m_t = self.model.compute_M_t(Sigma_0)
        
        v_t = np.zeros_like(m_t) 
        
        risks = []
        
        for t in range(self.T):
            bias_part = np.sum(self.L * m_t)
            variance_part = np.sum(self.L * v_t)
            
            risk = 0.5 * (bias_part + variance_part)
            self.risks = risk
            risks.append(risk)
            
            # Mises à jour pour l'itération t+1
            lr = self.get_step(t)
            At = self.get_P_matrix(t)
            
            # 1. Le biais continue d'être amorti
            m_t = At @ m_t
            
            # 2. La variance est amortie par Pt, ET on lui ajoute le nouveau bruit de l'étape t
            # (Hypothèse : on suppose que model.sigma est un vecteur contenant les sigma_{t,i})
            noise_t = (lr**2) * (self.model.sigma**2)
            v_t = At @ v_t + noise_t
            
        return np.array(risks)