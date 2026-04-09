import numpy as np
import matplotlib.pyplot as plt
from scheduled.schedules.base import ScheduleBase

class SGD:
    def __init__(self, sim, x0, schedule: ScheduleBase = None):
        self.sim = sim
        self.x0 = x0.reshape(-1, 1)
        self.list_At = {}
        self._schedule = schedule
        self.T = schedule._steps if schedule is not None else 100

        if self.sim.Lambda_vals is None:
            self.sim.compute_lambda()
        self.L = self.sim.Lambda_vals 

    def get_schedule(self):
        return self._schedule.schedule
    
    def get_step(self, t):
        assert self._schedule is not None, "Schedule is not defined"
        return self._schedule.schedule[t]

    def get_A_matrix(self, t):
        if t in self.list_At:
            return self.list_At[t]
        
        lr = self.get_step(t)
        term1 = np.diag((1 - lr * self.L)**2)
        term2 = np.diag(lr**2 * (self.L**2))
        term3 = (lr**2) * np.outer(self.L, self.L)
        self.list_At[t] = term1 + term2 + term3
        return self.list_At[t]

    def train(self, label="SGD", show=True):
        x = self.x0
        self.sim.n_samples = self.T
        Phi, Y = self.sim.generate_data()
        loss = []
        for t in range(self.T):
            phi, y = Phi[t].reshape(-1,1), Y[t]
            g = phi @ (np.dot(phi.T , x) - y)
            x = x - self.get_step(t) * g
            loss.append(self.sim.compute_theoretical_risk(x))
        if show:
            plt.plot(loss, label=label)
            plt.xlabel("Epoch")
            plt.ylabel("loss")
            plt.legend()
        return np.array(loss)


    def compute_theoretical_risk(self, t):
        """
        Compute the risk at step T efficiently in O(T).
        """
        assert t < self.T, "T must be less than the number of training steps"

        diff_0 = self.x0 - self.sim.x_star
        Sigma_0 = diff_0 @ diff_0.T
        _, m_t = self.sim.compute_M_t(Sigma_0)
        
        v_t = np.zeros(self.sim.dim)
        
        for i in range(t):
            lr = self.get_step(i)
            At = self.get_A_matrix(i)
            
            m_t = At @ m_t
            v_t = At @ v_t + (lr**2 * self.sim.sigma**2) * self.L

        bias_part = np.sum(self.L * m_t)
        variance_part = np.sum(self.L * v_t)
        irreducible_noise = self.sim.sigma**2
        
        return 0.5 * (bias_part + variance_part + irreducible_noise)
