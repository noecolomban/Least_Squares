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
        self.L = self.model.Lambda_vals
        self.risks = {}
        self.losses = None
  
    @property
    def T(self):
        return self._schedule._steps
    
    @property
    def schedule(self):
        return self._schedule
    
    @schedule.setter
    def schedule(self, schedule: ScheduleBase):
        self._schedule = schedule

    def get_schedule(self):
        return self._schedule.schedule
    
    def get_step(self, t):
        return self._schedule.schedule[t]

    def train(self, label="SGD", show=True):
        x = self.x0
        Phi, Y = self.model.generate_data(n_samples=self.T)
        loss = []
        for t in range(self.T):
            loss.append(self.model.compute_risk(x))

            phi, y = Phi[t].reshape(-1,1), Y[t]
            g = phi @ (np.dot(phi.T , x) - y)
            x = x - self.get_step(t) * g
        self.losses = np.array(loss)
        if show:
            plt.plot(self.losses, label=label)
            plt.xlabel("Epoch")
            plt.ylabel("loss")
            plt.legend()
        return self.losses
    

    def sample_slock(self, label="SGD", show=True, n_samples=1):
        loss = np.zeros((n_samples, self.T))
        for n in range(n_samples):
            x = self.x0
            Phi, Y = self.model.generate_slock(n_samples=self.T)
            for t in range(self.T):
                phi, y = Phi[t].reshape(-1,1), Y[t]
                g = phi @ (np.dot(phi.T , x) - y)
                x = x - self.get_step(t) * g
                loss[n, t] = self.model.compute_risk(x)
        self.losses = np.mean(loss, axis=0)
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
    def compute_all_theoretical_risks(self, separate_bias_variance=False) -> np.ndarray:
        pass



class SGD(BaseSGD):
    name = "True SGD"

    def __init__(self, model: LinearRegression, x0: np.ndarray, schedule: ScheduleBase):
        super().__init__(model, x0, schedule)



    def compute_all_theoretical_risks(self, separate_bias_variance=False) -> np.ndarray:
        """
        Calcule le risque théorique à chaque étape (Optimisé O(d))
        """
        diff_0 = self.x0 - self.model.x_star
        Sigma_0 = np.outer(diff_0, diff_0)
        
        _, m_t = self.model.compute_M_t(Sigma_0)
        m_t = m_t.flatten()
        
        v_t = np.zeros(self.model.dim) # variance
        risks = []

        if separate_bias_variance:
            biases = []
            variances = []

        irreducible_noise = self.model.sigma**2
        lambda_vec = self.L  # eigenvalues of H
        
        for t in range(self.T):
            bias_part = np.dot(lambda_vec, m_t)
            variance_part = np.dot(lambda_vec, v_t)
            risk = 0.5 * (bias_part + variance_part)
            
            self.risks[t] = risk
            risks.append(risk)
            
            if separate_bias_variance:
                biases.append(0.5 * bias_part)
                variances.append(0.5 * variance_part)
            if t < self.T - 1:
                lr = self.get_step(t)
                diag_part = (1 - lr * lambda_vec)**2 + (lr * lambda_vec)**2
                
                dot_m = np.dot(lambda_vec, m_t) 
                dot_v = np.dot(lambda_vec, v_t)  
                
                # update (noise is correctly injected into the variance here)
                m_t = diag_part * m_t + (lr**2) * lambda_vec * dot_m
                v_t = diag_part * v_t + (lr**2) * lambda_vec * dot_v + (lr**2 * irreducible_noise) * lambda_vec
        
        if separate_bias_variance:
            return np.array(biases), np.array(variances)
        else:
            return np.array(risks)
    

    def compute_all_slock_risks(self, batch=1, separate_bias_variance=False) -> np.ndarray:
        """
        Computes the theoretical risk at each step for the Slock model.
        Optimized for O(d) execution time using purely diagonal updates.
        """
        diff_0 = self.x0 - self.model.x_star
        Sigma_0 = np.outer(diff_0, diff_0)
        
        # Initialize bias vector (m_t)
        _, m_t = self.model.compute_M_t(Sigma_0)
        m_t = m_t.flatten()
        
        # Initialize variance vector (v_t) at zero
        v_t = np.zeros(self.model.dim)
        risks = []

        if separate_bias_variance:
            biases = []
            variances = []

        irreducible_noise = self.model.sigma**2
        lambda_vec = self.L  # Eigenvalues of the Hessian matrix (Lambda)
        
        # Slock core component: Trace of Lambda
        tr_Lambda = np.sum(lambda_vec)
        
        for t in range(self.T):
            # Calculate expected risks: 0.5 * lambda^T * state_vector
            bias_part = np.dot(lambda_vec, m_t)
            variance_part = np.dot(lambda_vec, v_t)
            risk = 0.5 * (bias_part + variance_part)
            
            self.risks[t] = risk
            risks.append(risk)
            
            if separate_bias_variance:
                biases.append(0.5 * bias_part)
                variances.append(0.5 * variance_part / batch)  # Adjust variance for batch size
                
            if t < self.T - 1:
                lr = self.get_step(t)
                
                # Slock state multiplier: I - 2*eta*Lambda + eta^2*tr(Lambda)*Lambda
                slock_factor = 1.0 - 2.0 * lr * lambda_vec + (lr**2) * tr_Lambda / batch * lambda_vec + (lr**2)*(1 - 1/batch) * lambda_vec**2
                
                # Decoupled Slock updates
                # Bias simply decays through the slock factor
                m_t = slock_factor * m_t
                
                # Variance decays through the slock factor and accumulates stochastic noise
                v_t = slock_factor * v_t + (lr**2 * irreducible_noise) * lambda_vec
        
        if separate_bias_variance:
            return np.array(biases), np.array(variances)
        else:
            return np.array(risks)
    
    #OPTIMIZED BY GEMINI
    def approx_all_theoretical_risks(self, separate_bias_variance=False, only_final_T=True) -> np.ndarray:
        # Initial setup
        diff_0 = self.x0 - self.model.x_star
        Sigma_0 = np.outer(diff_0, diff_0)
        _, m_0 = self.model.compute_M_t(Sigma_0)
        m_0 = m_0.flatten()
        sigma_sq = self.model.sigma**2
        
        # 1. Precompute learning rates to avoid calling the function 500,000 times
        lrs = np.array([self.get_step(t) for t in range(self.T)])[:, np.newaxis]
        L_row = self.L[np.newaxis, :]
        
        # 2. Precompute the entire update factors (U) and noise factors (S)
        print(f"T={self.T}, Precomputing update and noise factors for all steps...")
        U_matrix = (1 - lrs * L_row)**2 + 2 * (lrs * L_row)**2
        S_matrix = (lrs**2 * sigma_sq) * L_row
        
        # 3. Pre-allocate output arrays
        biases = np.zeros(self.T)
        variances = np.zeros(self.T)
        
        m_t = m_0.copy()
        v_t = np.zeros(self.model.dim)
        
        # 4. Ultra-fast loop using in-place operations (*= and +=) and np.dot
        print(f"T={self.T}, Computing risks using optimized loop...")
                
        for t in range(self.T):
            # np.dot is heavily optimized in C/BLAS and much faster than np.sum(A * B)
            biases[t] = 0.5 * np.dot(self.L, m_t)
            variances[t] = 0.5 * np.dot(self.L, v_t)
            
            # In-place updates: prevents Python from creating new arrays in memory on each loop iteration
            m_t *= U_matrix[t]
            v_t *= U_matrix[t]
            v_t += S_matrix[t]

        if separate_bias_variance:
            return biases, variances
        return biases + variances

    #GEMINI OPTIMIZED
    def approx_final_theoretical_risk_variable(self, separate_bias_variance=False) -> float | tuple[float, float]:
        # Initial setup
        diff_0 = self.x0 - self.model.x_star
        Sigma_0 = np.outer(diff_0, diff_0)
        _, m_T = self.model.compute_M_t(Sigma_0)
        m_T = m_T.flatten()
        
        v_T = np.zeros(self.model.dim)
        sigma_sq = self.model.sigma**2
        
        # Precompute learning rates as a 1D array to avoid function call overhead
        lrs = np.array([self.get_step(t) for t in range(self.T)])
        
        # Tight loop modifying the same memory slots at each step
        for lr in lrs:
            # Update factor for the current learning rate
            U = (1 - lr * self.L)**2 + 2 * (lr * self.L)**2
            
            # In-place updates (*= and +=) prevent memory allocation during the 500k steps
            m_T *= U
            v_T *= U
            v_T += (lr**2 * sigma_sq) * self.L
            
        # Compute final dot product only once at the very end
        final_bias = 0.5 * np.dot(self.L, m_T)
        final_variance = 0.5 * np.dot(self.L, v_T)
        
        if separate_bias_variance:
            return final_bias, final_variance
        return final_bias + final_variance
    


class NoisyGD(BaseSGD):
    name = "Noisy GD"

    def __init__(self, model: LinearRegression, x0: np.ndarray, schedule: ScheduleBase):
        super().__init__(model, x0, schedule)

    def compute_all_theoretical_risks(self) -> np.ndarray:
        diff_0 = self.x0 - self.model.x_star
        Sigma_0 = np.outer(diff_0, diff_0)
        _, m_t = self.model.compute_M_t(Sigma_0)
        
        v_t = np.zeros_like(m_t) 
        risks = []
        
        for t in range(self.T):
            bias_part = np.sum(self.L * m_t)
            variance_part = np.sum(self.L * v_t)
            
            risk = 0.5 * (bias_part + variance_part)
            self.risks[t] = risk
            risks.append(risk)
            
            # --- OPTIMISATION VECTORIELLE ---
            lr = self.get_step(t)
            
            # Vecteur amortisseur
            P_t_vector = (1 - lr * self.L)**2
            
            # 1. Element-wise update (simple '*' multiplication)
            m_t = P_t_vector * m_t
            
            # 2. Ajout du bruit
            noise_t = (lr**2) * (self.model.sigma**2) * self.L
            v_t = P_t_vector * v_t + noise_t
            
        return np.array(risks)