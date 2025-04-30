import torch
import utils


class OptionSet:
    """
    This class is used to price different types of options given the MonteCarlo paths
    produced by the current policy.
    We can choose any pricing method and any kind of derivative given the simulated underlying paths
    """
    def __init__(self, vanilla_calls, r, bermudan=None):
        """
        :param vanilla_calls: list of (strike, maturity) pairs
        :param bermudan: either a 4-tuple describing the Bermudan option
            (strike k₁ strike k₂ first-exercise t₁ final-maturity t₂) or None if we calibrate only to vanillas
        """
        self.calls = vanilla_calls
        self.bermudan = bermudan
        self.r = r

    def price_calls(self, paths, disc):
        """
        Monte-Carlo call prices for all strikes and maturities.
        :param paths: 2-D tensor
            first axis = time index 0…T
            second axis = path index 1…n
        :param disc: length T+1 vector with discount factors
        """
        # paths: (T+1, n) spot trajectories   disc = discount factors if needed
        t_axis, n = paths.shape
        prices = []
        for k, maturity in self.calls:
            # for every path, calculate the PayOff for each maturity and strike
            payoff = torch.clamp(paths[maturity] - k, min=0.)
            # calculate the price of the option as the mean of the payoffs for each MonteCarlo
            # path discounted to the present
            prices.append(payoff.mean() * disc[maturity])
        return torch.stack(prices)  # (n_calls,)

    def iv_calls(self, paths):
        S0 = paths[0, 0]
        ivs = []
        for K, mat in self.calls:
            pay = torch.clamp(paths[mat] - K, 0).mean()
            iv = utils.implied_vol_call(pay, S0, K, mat/252, self.r)
            ivs.append(iv)
        return torch.stack(ivs)

    # TODO: review pricing of bermudan option (it was chatGPT generated and I didn't review yet)
    def price_bermudan(self, paths, disc):
        if self.bermudan is None:                 # nothing to do
            return torch.tensor([], device=paths.device)
        k1, k2, t1, t2 = self.bermudan
        # Longstaff-Schwartz with polynomial basis (deg 2 for demo)
        spot_t1 = paths[t1]
        disc_t1 = disc[t1]
        spot_t2 = paths[t2]
        payoff_t2 = torch.clamp(spot_t2 - k2, min=0.)
        # regression of continuation at t1
        X = torch.stack([torch.ones_like(spot_t1), spot_t1, spot_t1**2], 1)
        coeff, _ = torch.lstsq((payoff_t2*disc[t2]/disc_t1).unsqueeze(1), X)
        cont_t1 = (X @ coeff).squeeze()
        ex_t1 = torch.clamp(spot_t1 - k1, min=0.)
        payoff = torch.where(ex_t1 >= cont_t1, ex_t1, payoff_t2*disc[t2]/disc_t1)
        return payoff.mean()                          # scalar tensor
