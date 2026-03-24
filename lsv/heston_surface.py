import numpy as np
from scipy.integrate import quad
from lsv.utils import implied_volatility

def heston_char_func(u, T, S0, K, r, q, kappa, theta, sigma, rho, v0):
    F = S0 * np.exp((r - q) * T)
    x = np.log(F / K)

    d = np.sqrt((kappa - rho * sigma * 1j * u)**2 + sigma**2 * (u**2 + 1j * u))
    g = (kappa - rho * sigma * 1j * u - d) / (kappa - rho * sigma * 1j * u + d)
    exp_dT = np.exp(-d * T)

    C = (kappa * theta / sigma**2) * (
        (kappa - rho * sigma * 1j * u - d) * T
        - 2 * np.log((1 - g * exp_dT) / (1 - g))
    )
    D = ((kappa - rho * sigma * 1j * u - d) / sigma**2) * (
        (1 - exp_dT) / (1 - g * exp_dT)
    )
    return np.exp(C + D * v0 + 1j * u * x)


def heston_price(S0, K, T, r, q, kappa, theta, sigma, rho, v0, option_type='call'):
    def integrand_P1(u):
        phi   = heston_char_func(u - 1j, T, S0, K, r, q, kappa, theta, sigma, rho, v0)
        phi_0 = heston_char_func(-1j,    T, S0, K, r, q, kappa, theta, sigma, rho, v0)
        return np.real(phi / (phi_0 * 1j * u))

    def integrand_P2(u):
        phi = heston_char_func(u, T, S0, K, r, q, kappa, theta, sigma, rho, v0)
        return np.real(phi / (1j * u))

    P1 = 0.5 + (1/np.pi) * quad(integrand_P1, 1e-4, 100, limit=500)[0]
    P2 = 0.5 + (1/np.pi) * quad(integrand_P2, 1e-4, 100, limit=500)[0]

    call = S0 * np.exp(-q*T) * P1 - K * np.exp(-r*T) * P2

    if option_type == 'call':
        return call
    elif option_type == 'put':
        return call - S0*np.exp(-q*T) + K*np.exp(-r*T)


def generate_iv_surface(S0, r, q, kappa, theta, sigma, rho, v0, moneyness_grid, T_grid):
    IV_surface = np.zeros((len(moneyness_grid), len(T_grid)))

    for i, m in enumerate(moneyness_grid):
        for j, T in enumerate(T_grid):
            K = m * S0
            price = heston_price(S0, K, T, r, q, kappa, theta, sigma, rho, v0, 'call')
            iv = implied_volatility(S0, K, T, r, q, price, 'call')
            IV_surface[i, j] = iv if iv is not None else np.nan

    return IV_surface

