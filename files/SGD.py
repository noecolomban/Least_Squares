from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt

class SGD(ABC):
    def __init__(self, sim, x0):
        self.sim = sim
        self.x0 = x0.reshape(-1, 1)
        self.list_At = {}

        if self.sim.Lambda_vals is None:
            self.sim.compute_lambda()
        self.L = self.sim.Lambda_vals 

    @abstractmethod
    def get_step(self, t):
        pass

    def get_A_matrix(self, t):
        if t in self.list_At:
            return self.list_At[t]
        
        lr = self.get_step(t)
        
        term1 = np.diag((1 - lr * self.L)**2)
        
        term2 = np.diag(lr**2 * (self.L**2))
        
        term3 = (lr**2) * np.outer(self.L, self.L)

        self.list_At[t] = term1 + term2 + term3
        return self.list_At[t]

    def train(self, T=100, label="SGD", show=True):
        x = self.x0
        self.sim.n_samples = T
        Phi, Y = self.sim.generate_data()
        loss = []
        for t in range(T):
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


    def compute_theoretical_risk(self, T):
        """
        Compute the risk at step T+1 efficiently in O(T).
        """
        diff_0 = self.x0 - self.sim.x_star
        Sigma_0 = diff_0 @ diff_0.T
        _, m_t = self.sim.compute_M_t(Sigma_0)
        
        v_t = np.zeros(self.sim.dim)
        
        for t in range(T + 1):
            lr = self.get_step(t)
            At = self.get_A_matrix(t)
            
            m_t = At @ m_t
            v_t = At @ v_t + (lr**2 * self.sim.sigma**2) * self.L

        bias_part = np.sum(self.L * m_t)
        variance_part = np.sum(self.L * v_t)
        irreducible_noise = self.sim.sigma**2
        
        return 0.5 * (bias_part + variance_part + irreducible_noise)
    


class SGD_poly(SGD):
    def __init__(self, sim, x0, eta, gamma):
        super().__init__(sim, x0)
        self.eta = eta
        self.gamma = gamma

    def get_step(self, t):
        return self.eta / (t + 1)**self.gamma
    
    def plot(self, T=100):
        X = np.arange(T)
        Y = [self.compute_theoretical_risk(t) for t in range(T)]
        plt.plot(X, Y, label = rf"Theory $\gamma$ = {self.gamma:.2f}", linewidth=3)
        plt.title(fr"$\eta_t = \frac{{\eta}}{{t^\gamma}}$, $\gamma = {self.gamma}$, $\eta = {self.eta}$")
        plt.xlabel("iteration")
        plt.ylabel("loss")
        plt.legend()


class SGD_wsd(SGD):
    def __init__(self, sim, x0, T0, T, eta):
        super().__init__(sim, x0)
        self.T0 = T0
        self.T = T
        self.eta = eta
    
    def get_step(self, t):
            if t < self.T0:
                 return self.eta
            else:
                return max(0,self.eta *(self.T + 1 - t) / (self.T - self.T0 + 1))
    
    def plot(self, T=100):
        X = np.arange(T)
        Y = [self.compute_theoretical_risk(t) for t in range(T)]
        plt.plot(X, Y, label = rf"Theory $T0/T$ = {self.T0/self.T:.2f}", linewidth=3)
        plt.title(rf"$T0 = {self.T0}$, $\eta = {self.eta}$")
        plt.xlabel("iteration")
        plt.ylabel("loss")
        plt.legend()