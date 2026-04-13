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
        return self._schedule.schedule[t]

    def train(self, label="SGD", show=True):
        x = self.x0
        Phi, Y = self.model.generate_data(n_samples=self.T)
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

    def compute_all_theoretical_risks(self) -> np.ndarray:
        diff_0 = self.x0 - self.model.x_star
        Sigma_0 = diff_0 @ diff_0.T
        _, m_t = self.model.compute_M_t(Sigma_0)
        
        v_t = np.zeros(self.model.dim)
        risks = []
        irreducible_noise = self.model.sigma**2
        
        for t in range(self.T):
            # Le calcul du risque reste identique
            bias_part = np.sum(self.L * m_t)
            variance_part = np.sum(self.L * v_t)
            risk = 0.5 * (bias_part + variance_part + irreducible_noise)
            self.risks[t] = risk
            risks.append(risk)
            
            # --- OPTIMISATION O(d) ICI ---
            lr = self.get_step(t)
            
            # Vecteur diagonal (diag_part est de taille d)
            diag_part = (1 - lr * self.L)**2 + (lr * self.L)**2
            
            # Produit scalaire (scalaire = O(d))
            dot_L_mt = np.dot(self.L, m_t)
            dot_L_vt = np.dot(self.L, v_t)
            
            # Mise à jour purement vectorielle (O(d) au lieu de O(d^2))
            m_t = diag_part * m_t + (lr**2) * self.L * dot_L_mt
            v_t = diag_part * v_t + (lr**2) * self.L * dot_L_vt + (lr**2 * self.model.sigma**2) * self.L
        
        return np.array(risks)
    
    def approx_all_theoretical_risks(self) -> np.ndarray:
        diff_0 = self.x0 - self.model.x_star
        Sigma_0 = diff_0 @ diff_0.T
        _, m_t = self.model.compute_M_t(Sigma_0)
        
        v_t = np.zeros(self.model.dim)
        risks = []
        irreducible_noise = self.model.sigma**2
        
        for t in range(self.T):
            # Le calcul du risque reste identique
            bias_part = np.sum(self.L * m_t)
            variance_part = np.sum(self.L * v_t)
            risk = 0.5 * (bias_part + variance_part + irreducible_noise)
            self.risks[t] = risk
            risks.append(risk)
            
            lr = self.get_step(t)
            
            # Vecteur diagonal de base
            diag_part = (1 - lr * self.L)**2 + (lr * self.L)**2
            
            # --- L'APPROXIMATION EST ICI ---
            # Au lieu de faire un produit scalaire global (qui couple toutes les dimensions),
            # on fait une simple multiplication élément par élément (L^2 * m_t)
            approx_term_mt = (self.L**2) * m_t
            approx_term_vt = (self.L**2) * v_t
            
            # Mise à jour totalement découplée (chaque dimension i vit sa vie de son côté)
            m_t = diag_part * m_t + (lr**2) * approx_term_mt
            v_t = diag_part * v_t + (lr**2) * approx_term_vt + (lr**2 * self.model.sigma**2) * self.L
        
        return np.array(risks)



class NoisyGD(BaseSGD):
    name = "Noisy GD"

    def __init__(self, model: LinearRegression, x0: np.ndarray, schedule: ScheduleBase):
        super().__init__(model, x0, schedule)

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
            # self.risks = risk  # <-- Attention: Ceci était un bug dans votre code originel (écrasement du dict), je l'ai corrigé en dessous
            self.risks[t] = risk
            risks.append(risk)
            
            # --- OPTIMISATION VECTORIELLE ---
            lr = self.get_step(t)
            
            # Vecteur amortisseur
            P_t_vector = (1 - lr * self.L)**2
            
            # 1. Mise à jour élément par élément (multiplication simple '*')
            m_t = P_t_vector * m_t
            
            # 2. Ajout du bruit
            noise_t = (lr**2) * (self.model.sigma**2)
            v_t = P_t_vector * v_t + noise_t
            
        return np.array(risks)