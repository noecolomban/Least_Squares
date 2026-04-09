import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, dim=5, sigma=0.1, n_samples=1000):
        """      
        :param dim: Dimension des vecteurs x (d)
        :param sigma: Écart-type du bruit epsilon
        :param n_samples: Nombre d'échantillons (N)
        """
        self.dim = dim
        self.sigma = sigma
        self.n_samples = n_samples
        
        self.H = self._generate_spd_matrix(dim)
        self.compute_lambda()
        self.w_star = np.random.randn(dim, 1)
        
        self.X = None
        self.Y = None
        self.w_hat = None

    def _generate_spd_matrix(self, d):
        """Génère une matrice H telle que H = QΛQ^T > 0"""
        A = np.random.randn(d, d)
        H = A @ A.T + 0.1 * np.eye(d)
        return H

    def generate_data(self):
        """Génère les échantillons {(xi, yi)} selon x ~ N(0, H) et y = x.T*w* + eps"""
        self.X = np.random.multivariate_normal(
            np.zeros(self.dim), self.H, size=self.n_samples
        )
        
        epsilon = np.random.normal(0, self.sigma, size=(self.n_samples, 1))
        
        self.Y = self.X @ self.w_star + epsilon
        return self.X, self.Y

    def fit(self):
        """Calcule l'estimateur des moindres carrés (w_hat)"""
        if self.X is None:
            self.generate_data()
            
        self.w_hat, _, _, _ = np.linalg.lstsq(self.X, self.Y, rcond=None)
        return self.w_hat

    def compute_empirical_risk(self):
        """Calcule R_emp = 1/2 * moyenne((X*w - Y)^2)"""
        if self.w_hat is None:
            self.fit()
        
        predictions = self.X @ self.w_hat
        risk = 0.5 * np.mean((predictions - self.Y)**2)
        return risk

    def compute_theoretical_risk(self, w=None):
        """
        Calcule R(w) = 1/2 * E[(<w, x> - y)^2] 
        Analytiquement : 1/2 * (w - w*)^T H (w - w*) + 1/2 * sigma^2
        """
        if w is None:
            w = self.w_hat
            
        diff_w = w - self.w_star
        estimation_error = 0.5 * (diff_w.T @ self.H @ diff_w)
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

    def compute_Sigma_t(self, list_of_w_t):
        """
        Calcule la matrice de covariance empirique Sigma_t à partir d'une liste
        de vecteurs de poids (itérés) obtenus à l'étape t.
        Sigma_t = E[(w_t - w*)(w_t - w*)^T]
        """
        W = np.array(list_of_w_t).reshape(-1, self.dim)
        
        diff = W - self.w_star.T  # Broadcasting sur toutes les simulations
        
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