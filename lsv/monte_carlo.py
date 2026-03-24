import numpy as np

def simulate_lsv_paths(S0, r, q, kappa, theta, sigma, rho, v0, spline_L, T_grid, N_paths=10000):

    N_steps = len(T_grid)
    S_paths = np.zeros((N_paths, N_steps))
    V_paths = np.zeros((N_paths, N_steps))
    
    S_paths[:, 0] = S0
    V_paths[:, 0] = v0
    
    for j in range(1, N_steps):
        dt = T_grid[j] - T_grid[j-1]
        t_prev = T_grid[j-1]
        
        Z1 = np.random.standard_normal(N_paths)
        Z2 = rho * Z1 + np.sqrt(1.0 - rho**2) * np.random.standard_normal(N_paths)
        
        V_prev = V_paths[:, j-1]
        S_prev = S_paths[:, j-1]
        
        V_plus = np.maximum(V_prev, 0.0)
        
        V_next = V_prev + kappa * (theta - V_plus) * dt + sigma * np.sqrt(V_plus * dt) * Z2
        V_paths[:, j] = V_next
        
        m_prev = S_prev / S0
        L_vals = spline_L.ev(m_prev, np.full(N_paths, t_prev))
        
        L_vals = np.clip(L_vals, 0.05, 5.0) 
        vol_step = L_vals * np.sqrt(V_plus)
        S_paths[:, j] = S_prev * np.exp((r - q - 0.5 * vol_step**2) * dt + vol_step * np.sqrt(dt) * Z1)
        
    return S_paths


def price_option_lsv(S0, K, T, r, q, kappa, theta, sigma, rho, v0, spline_L, T_grid, option_type='call', N_paths=10000):
    idx_T = np.abs(T_grid - T).argmin()
    
    S_paths = simulate_lsv_paths(S0, r, q, kappa, theta, sigma, rho, v0, spline_L, T_grid, N_paths)
    
    S_T = S_paths[:, idx_T]
    
    if option_type == 'call':
        payoff = np.maximum(S_T - K, 0.0)
    elif option_type == 'put':
        payoff = np.maximum(K - S_T, 0.0)
    else:
        raise ValueError("NAN")
        
    price = np.exp(-r * T) * np.mean(payoff)
    
    return price

