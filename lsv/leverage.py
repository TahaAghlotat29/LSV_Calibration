import numpy as np
from scipy.interpolate import RectBivariateSpline

def simulate_heston_paths(S0, r, q, kappa, theta, sigma, rho, v0, T_grid, N_paths=10000):
    """
    Simulates Heston model paths using the Euler-Maruyama scheme.
    Uses the 'Full Truncation' scheme for the variance process to handle negative values.
    """
    N_steps = len(T_grid)
    S_paths = np.zeros((N_paths, N_steps))
    V_paths = np.zeros((N_paths, N_steps))
    
    S_paths[:, 0] = S0
    V_paths[:, 0] = v0
    
    for j in range(1, N_steps):
        dt = T_grid[j] - T_grid[j-1]
        
        Z1 = np.random.standard_normal(N_paths)
        Z2 = rho * Z1 + np.sqrt(1.0 - rho**2) * np.random.standard_normal(N_paths)
        
        V_prev = V_paths[:, j-1]
        S_prev = S_paths[:, j-1]
        
        V_plus = np.maximum(V_prev, 0.0)
        V_next = V_prev + kappa * (theta - V_plus) * dt + sigma * np.sqrt(V_plus * dt) * Z2
        V_paths[:, j] = V_next  
        
        S_paths[:, j] = S_prev * np.exp((r - q - 0.5 * V_plus) * dt + np.sqrt(V_plus * dt) * Z1)
        
    V_paths = np.maximum(V_paths, 1e-8)
    
    return S_paths, V_paths

def compute_leverage_function(local_vol_surface, S_paths, V_paths, moneyness_grid, T_grid, S0):
    """
    Computes the leverage function L(K, t) using a Nadaraya-Watson kernel estimator 
    to calculate the conditional expectation E[V_t | S_t = K].
    """
    K_grid = moneyness_grid * S0
    L = np.zeros_like(local_vol_surface)
    
    for j in range(1, len(T_grid)):
        S_j = S_paths[:, j]
        V_j = V_paths[:, j]
        std_S = np.std(S_j)
        h = 1.06 * std_S * (len(S_j) ** (-0.2))
        h = max(h, 1e-4) 
        
        for i, K in enumerate(K_grid):
            weights = np.exp(-0.5 * ((S_j - K) / h) ** 2)
            sum_weights = np.sum(weights)
            
            if sum_weights > 1e-8:
                EV_cond = np.sum(weights * V_j) / sum_weights
            else:
                EV_cond = np.mean(V_j) 
            
            EV_cond = max(EV_cond, 1e-8)
            
            L[i, j] = local_vol_surface[i, j] / np.sqrt(EV_cond)
            
    L[:, 0] = L[:, 1]
    
    L = np.clip(L, 0.05, 5.0)
    
    spline_L = RectBivariateSpline(moneyness_grid, T_grid, L, kx=3, ky=3)
    
    return L, spline_L