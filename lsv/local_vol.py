import numpy as np

class SyntheticMarketSurface:
    """
    Generates a synthetic implied volatility surface via SSVI 
    and calculates the corresponding local volatility using Dupire's formula in terms of total variance.
    """
    def __init__(self, S0, r, q, rho=-0.6, eta=0.5, gamma=0.5):
        self.S0 = S0
        self.r = r
        self.q = q
        self.rho = rho
        self.eta = eta
        self.gamma = gamma
        
    def _theta(self, t):
        return 0.04 * t 
        
    def _phi(self, theta):
        return self.eta / (theta ** self.gamma)

    def total_variance(self, k, t):
        """ Calculates the total variance w(k, t) according to the SSVI model """
        theta_t = self._theta(t)
        phi_t = self._phi(theta_t)
        
        sqrt_term = np.sqrt((phi_t * k + self.rho)**2 + (1 - self.rho**2))
        w = 0.5 * theta_t * (1 + self.rho * phi_t * k + sqrt_term)
        return w

    def implied_vol(self, K, T):
        F = self.S0 * np.exp((self.r - self.q) * T)
        k = np.log(K / F)
        w = self.total_variance(k, T)
        return np.sqrt(w / T)

    def local_vol_dupire(self, K, T, dk=1e-4, dt=1e-4):
        """ Calculates Dupire's local volatility using finite differences on w(k,t) """
        F = self.S0 * np.exp((self.r - self.q) * T)
        k = np.log(K / F)
        
        T = np.maximum(T, 1e-5) 
        
        w = self.total_variance(k, T)
        
        dw_dt = (self.total_variance(k, T + dt) - w) / dt
        
        w_plus = self.total_variance(k + dk, T)
        w_minus = self.total_variance(k - dk, T)
        
        dw_dk = (w_plus - w_minus) / (2 * dk)
        d2w_dk2 = (w_plus - 2*w + w_minus) / (dk**2)
        
        term1 = 1.0 - (k / w) * dw_dk
        term2 = 0.25 * (-0.25 - 1.0/w + (k**2)/(w**2)) * (dw_dk**2)
        term3 = 0.5 * d2w_dk2
        
        denominator = term1 + term2 + term3
        
        denominator = np.maximum(denominator, 1e-6)
        local_var = dw_dt / denominator
        
        return np.sqrt(np.maximum(local_var, 1e-6))
    



