import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, dim=5, sigma=0.1, n_samples=1000):
        """      
        :param dim: Dimension of phi vectors (d)
        :param sigma: Standard deviation of noise epsilon
        :param n_samples: Number of samples (N)
        """
        self.dim = dim
        self.sigma = sigma
        self.n_samples = n_samples
        
        self.H = self._generate_spd_matrix(dim)
        self.compute_lambda()
        self.x_star = np.random.randn(dim, 1)
        
        self.phi = None
        self.Y = None
        self.x_hat = None

    def _generate_spd_matrix(self, d):
        """Generate a matrix H such that H = QΛQ^T > 0"""
        A = np.random.randn(d, d)
        H = A @ A.T + 0.1 * np.eye(d)
        return H

    def generate_data(self):
        """Generate samples {(phi_i, y_i)} according to phi ~ N(0, H) and y = phi.T*x* + eps"""
        self.phi = np.random.multivariate_normal(
            np.zeros(self.dim), self.H, size=self.n_samples
        )
        
        epsilon = np.random.normal(0, self.sigma, size=(self.n_samples, 1))
        
        self.Y = self.phi @ self.x_star + epsilon
        return self.phi, self.Y

    def fit(self):
        """Compute the least squares estimator (x_hat)"""
        if self.phi is None:
            self.generate_data()
            
        self.x_hat, _, _, _ = np.linalg.lstsq(self.phi, self.Y, rcond=None)
        return self.x_hat

    def compute_empirical_risk(self):
        """Compute R_emp = 1/2 * mean((Phi*x - Y)^2)"""
        if self.x_hat is None:
            self.fit()
        
        predictions = self.phi @ self.x_hat
        risk = 0.5 * np.mean((predictions - self.Y)**2)
        return risk

    def compute_theoretical_risk(self, x=None):
        """
        Compute R(x) = 1/2 * E[(<x, phi> - y)^2] 
        Analytically: 1/2 * (x - x*)^T H (x - x*) + 1/2 * sigma^2
        """
        if x is None:
            x = self.x_hat
            
        diff_x = x - self.x_star
        estimation_error = 0.5 * (diff_x.T @ self.H @ diff_x)
        noise_floor = 0.5 * (self.sigma**2)
        
        total_risk = estimation_error + noise_floor
        return total_risk[0]
    
    def compute_lambda(self):
        """
        Compute the eigendecomposition of H such that H = Q * Lambda * Q^T.
        Return the diagonal matrix Lambda.
        """
        self.Lambda_vals, self.Q = np.linalg.eigh(self.H)
        
        Lambda_matrix = np.diag(self.Lambda_vals)
        self.Lambda = Lambda_matrix
        self.Lambda_vals = self.Lambda_vals
        return Lambda_matrix

    def get_restriction(self, i, j):
        """
        Compute Lambda_{i:j} as defined in the text: diag(lambda_i, ..., lambda_j).
        Note: Indices i and j follow mathematical notation (starting at 0 or 1 depending on context).
        """
        if self.Lambda_vals is None:
            self.compute_lambda()
            
        subset = self.Lambda_vals[i:j+1]
        return np.diag(subset)

    def compute_Sigma_t(self, list_of_x_t):
        """
        Compute the empirical covariance matrix Sigma_t from a list
        of weight vectors (iterates) obtained at step t.
        Sigma_t = E[(x_t - x*)(x_t - x*)^T]
        """
        W = np.array(list_of_x_t).reshape(-1, self.dim)
        
        diff = W - self.x_star.T
        
        n_sims = len(list_of_x_t)
        Sigma_t = (diff.T @ diff) / n_sims
        return Sigma_t

    def compute_M_t(self, Sigma_t):
        """
        Compute Mt = Q * Sigma_t * Q^T (the covariance in the eigenvector space)
        and mt = diag(Mt).
        """
        if self.Q is None:
            self.compute_lambda()
            
        Mt = self.Q.T @ Sigma_t @ self.Q
 
        mt = np.diag(Mt)
        
        return Mt, mt