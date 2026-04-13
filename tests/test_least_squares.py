import pytest
import numpy as np
from src.least_squares import LinearRegression, PowerLawRegression

@pytest.fixture
def base_model():
    """Fixture pour instancier un modèle standard avant chaque test."""
    return LinearRegression(dim=5, sigma=0.1, n_samples=100)

def test_linear_regression_initialization(base_model):
    assert base_model.dim == 5
    assert base_model.H.shape == (5, 5)
    assert base_model.x_star.shape == (5, 1)

def test_generate_data(base_model):
    phi, Y = base_model.generate_data()
    assert phi.shape == (100, 5)
    assert Y.shape == (100, 1)

def test_power_law_eigenvalues():
    dim = 5
    exponent = 0.5
    model = PowerLawRegression(dim=dim, exponent=exponent)
    
    # On vérifie que les valeurs propres décroissent bien selon la loi de puissance
    expected_lambda = [1.0 / (i**exponent) for i in range(1, dim + 1)]
    
    # np.allclose gère les petites imprécisions flottantes
    assert np.allclose(sorted(model.Lambda_vals, reverse=True), expected_lambda)