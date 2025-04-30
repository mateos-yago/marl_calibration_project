import torch
from scipy.stats import norm
from torch.distributions import Normal


def european_bs_price(S, K, T, sigma, call=True, r=0.0, dtype=torch.float32, device=None):
    """Black–Scholes price (scalar tensors accepted)."""

    S = torch.as_tensor(S, dtype=dtype, device=device)
    K = torch.as_tensor(K, dtype=dtype, device=device)
    T = torch.as_tensor(T, dtype=dtype, device=device)
    sigma = torch.as_tensor(sigma, dtype=dtype, device=device)
    r = torch.as_tensor(r, dtype=dtype, device=device)

    sqrt_T = torch.sqrt(T)
    d1 = (torch.log(S / K) + 0.5 * sigma.pow(2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    df = torch.exp(-r * T)

    # torch-native standard normal CDF (differentiable & GPU-friendly)
    normal = Normal(0.0, 1.0)
    if call:
        price = S * normal.cdf(d1) - K * df * normal.cdf(d2)
    else:
        price = K * df * normal.cdf(-d2) - S * normal.cdf(-d1)

    return price


def implied_vol_call(price, S, K, T, r=0.0, tol=1e-6):
    """Scalar IV via bisection (enough for training-time reward)."""
    low, high = 1e-4, 5.0
    for _ in range(50):
        mid = 0.5 * (low + high)
        calc = european_bs_price(S, K, T, mid, True, r)
        high = torch.where(calc > price, mid, high)
        low = torch.where(calc > price, low, mid)
    return 0.5 * (low + high)
