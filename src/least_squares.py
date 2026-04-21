import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, dim=5, sigma=0.1, n_samples=1000, H: None | np.ndarray = None):
        """      
        :param dim: Dimension of phi vectors (d)
        :param sigma: Standard deviation of noise epsilon
        :param n_samples: Number of samples (N)
        :param H: The covariance matrix for the phi vectors
        """
        self.dim = dim
        self.sigma = sigma
        self.n_samples = n_samples
        
        if H is not None:
            assert H.shape == (dim, dim), "H must be a square matrix of shape (dim, dim)"
            self.H = H
        else:
            self.H = self._generate_spd_matrix(dim)
        
        self.Lambda, self.Lambda_vals, self.Q = self.compute_lambda()
        self.x_star = np.random.randn(dim, 1)
        
        self.phi = None
        self.Y = None
        self.x_hat = None

    def _generate_spd_matrix(self, d):
        """Generate a matrix H such that H = QΛQ^T > 0"""
        A = np.random.randn(d, d)
        H = A @ A.T + 0.1 * np.eye(d)
        return H

    def generate_data(self, n_samples=None):
        if n_samples is None:
            n_samples = self.n_samples
        """Generate samples {(phi_i, y_i)} according to phi ~ N(0, H) and y = phi.T*x* + eps"""
        self.phi = np.random.multivariate_normal(
            np.zeros(self.dim), self.H, size=n_samples
        )
        
        epsilon = np.random.normal(0, self.sigma, size=(n_samples, 1))
        
        self.Y = self.phi @ self.x_star + epsilon
        return self.phi, self.Y

    def fit(self):
        """Compute the least squares estimator (x_hat)"""
        if self.phi is None:
            self.generate_data()
            
        self.x_hat, _, _, _ = np.linalg.lstsq(self.phi, self.Y, rcond=None)
        return self.x_hat


    def compute_risk(self, x):
        """
        Compute R(x)-R* = 1/2 * E[(<x, phi> - y)^2] - 1/2 * E[(<x*, phi> - y)^2]
        Analytically: 1/2 * (x - x*)^T H (x - x*) 
        """
        diff_x = x - self.x_star
        total_risk = 0.5 * (diff_x.T @ self.H @ diff_x)
        return total_risk[0]
    
    def compute_lambda(self):
        Lambda_vals, Q = np.linalg.eigh(self.H)
        # Sort in descending order
        idx = np.argsort(Lambda_vals)[::-1]
        Lambda_vals = Lambda_vals[idx]
        Q = Q[:, idx]
        
        Lambda_matrix = np.diag(Lambda_vals)
        return Lambda_matrix, Lambda_vals, Q

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
    
    @property
    def delta0(self):
        """Compute delta0: x0 - x* in the eigenvector space"""
        if self.Q is None:
            self.compute_lambda()
        return self.Q.T @ (self.x0 - self.x_star)


class PowerLawRegression(LinearRegression):
    def __init__(self, dim=5, sigma=0.1, n_samples=1000, exponent=0.5):
        H = self._generate_power_law_H(dim, exponent)
        super().__init__(dim, sigma, n_samples, H)
        self.exponent = exponent

    @staticmethod
    def _generate_power_law_H(dim, exponent):
        """Generate a matrix H with eigenvalues that decay as a power law: lambda_i = 1/i^exponent"""
        A = np.random.randn(dim, dim)
        Q, _ = np.linalg.qr(A)  # Orthonormalize to get Q
        Lambda_vals = np.array([1.0 / (i**exponent) for i in range(1, dim + 1)])
        Lambda_matrix = np.diag(Lambda_vals)
        H = Q @ Lambda_matrix @ Q.T
        return H
    

def compute_power_x0(dim, x_star, Q, beta=1):
    """Compute an initial point x0 such that delta0 = 1/i**beta in the eigenvector space"""
    delta0 = np.array([1.0 / (i**beta) for i in range(1, dim + 1)])
    x0 = x_star + Q @ delta0
    return x0