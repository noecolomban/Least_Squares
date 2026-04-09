import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, dim=5, sigma=0.1, n_samples=1000):
        """      
        :param dim: Dimension des vecteurs phi (d)
        :param sigma: Écart-type du bruit epsilon
        :param n_samples: Nombre d'échantillons (N)
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
        """Génère une matrice H telle que H = QΛQ^T > 0"""
        A = np.random.randn(d, d)
        H = A @ A.T + 0.1 * np.eye(d)
        return H

    def generate_data(self):
        """Génère les échantillons {(xi, yi)} selon phi ~ N(0, H) et y = phi.T*x* + eps"""
        self.phi = np.random.multivariate_normal(
            np.zeros(self.dim), self.H, size=self.n_samples
        )
        
        epsilon = np.random.normal(0, self.sigma, size=(self.n_samples, 1))
        
        self.Y = self.phi @ self.x_star + epsilon
        return self.phi, self.Y

    def fit(self):
        """Calcule l'estimateur des moindres carrés (x_hat)"""
        if self.phi is None:
            self.generate_data()
            
        self.x_hat, _, _, _ = np.linalg.lstsq(self.phi, self.Y, rcond=None)
        return self.x_hat

    def compute_empirical_risk(self):
        """Calcule R_emp = 1/2 * moyenne((Phi*x - Y)^2)"""
        if self.x_hat is None:
            self.fit()
        
        predictions = self.phi @ self.x_hat
        risk = 0.5 * np.mean((predictions - self.Y)**2)
        return risk

    def compute_theoretical_risk(self, x=None):
        """
        Calcule R(x) = 1/2 * E[(<x, phi> - y)^2] 
        Analytiquement : 1/2 * (x - x*)^T H (x - x*) + 1/2 * sigma^2
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
        Calcule l'eigendecomposition de H tel que H = Q * Lambda * Q^T.
        Retourne la matrice diagonale Lambda.
        """
        # eigenvalues sont retournés dans l'ordre croissant par défaut
        self.Lambda_vals, self.Q = np.linalg.eigh(self.H)
        
        Lambda_matrix = np.diag(self.Lambda_vals)
        self.Lambda = Lambda_matrix
        self.Lambda_vals = self.Lambda_vals
        return Lambda_matrix

    def get_restriction(self, i, j):
        """
        Calcule Lambda_{i:j} comme défini dans le texte : diag(lambda_i, ..., lambda_j).
        Note : Les indices i et j suivent la notation mathématique (commençant à 0 ou 1 selon le contexte).
        """
        if self.Lambda_vals is None:
            self.compute_lambda()
            
        subset = self.Lambda_vals[i:j+1]
        return np.diag(subset)

    def compute_Sigma_t(self, list_of_x_t):
        """
        Calcule la matrice de covariance empirique Sigma_t à partir d'une liste
        de vecteurs de poids (itérés) obtenus à l'étape t.
        Sigma_t = E[(x_t - x*)(x_t - x*)^T]
        """
        W = np.array(list_of_x_t).reshape(-1, self.dim)
        
        diff = W - self.x_star.T  # Broadcasting sur toutes les simulations
        
        n_sims = len(list_of_w_t)
        Sigma_t = (diff.T @ diff) / n_sims
        return Sigma_t

    def compute_M_t(self, Sigma_t):
        """
        Calcule Mt = Q * Sigma_t * Q^T (la covariance dans l'espace des vecteurs propres)
        et mt = diag(Mt).
        """
        if self.Q is None:
            self.compute_lambda()
            
        Mt = self.Q.T @ Sigma_t @ self.Q # Rotation dans la base des vecteurs propres
 
        mt = np.diag(Mt)
        
        return Mt, mt