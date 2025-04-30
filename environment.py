import torch
from sklearn.neighbors import KNeighborsRegressor  # basis-player interp
from options_set import OptionSet
import utils


EPS = 1e-8


class MARLVolEnv:
    def __init__(self, n_agents, S0, T, dt, option_set: OptionSet, market_iv, n_basis=100, device='cpu'):
        """
        :param n_agents: number of agents (paths)
        :param T: max time to maturity
        :param dt: time step, expressed as a fraction of T
        :param option_set: OptionSet object
        :param n_basis: number of basis agents
        :param device: cuda device
        """
        self._prev_E = None
        self.S0 = S0
        self.market_iv = torch.as_tensor(market_iv, dtype=torch.float32, device=device)
        self.n_agents, self.T = n_agents, T
        self.dt = torch.as_tensor(dt, dtype=torch.float32, device=device)
        self.device = device
        self.opts = option_set
        self.n_basis = n_basis
        # constant discount factor here (r=0)
        self.discount = torch.ones(T + 1, device=device)
        # reusable knn interpolator for basis-noise → all paths
        self.knn = KNeighborsRegressor(n_neighbors=min(8, n_basis))

    @staticmethod
    def _safe(tensor: torch.Tensor) -> torch.Tensor:
        """Clamp to keep spot > 0 before log."""
        return torch.clamp(tensor, min=EPS)

    def reset(self):
        """
        Prepare a fresh episode:
        • zero the telescoping-error accumulator  (_prev_E)
        """
        self._prev_E = torch.zeros(1, device=self.device)  #  E_{-1}=0

    def build_state(self, t, spot, log_sigma, spot_t1=None, log_sigma_t1=None):
        """
        Bare-bones state: (t_norm, logS, logσ)
        Extend as needed.  For Bermudan, we optionally include (S_{t1}, σ_{t1}).
        """
        time_tensor = torch.full((self.n_agents,), t / self.T, device=self.device)
        state = torch.stack([time_tensor, torch.log(self._safe(spot)), log_sigma], 1)
        # TODO: Include history of past observations using LSTM encoding?

        # FOR NOW, IGNORE THIS PART (IT IS FOR BERMUDAN OPTION EXPERIMENT)
        if spot_t1 is not None:
            extra = torch.stack([torch.log(self._safe(spot_t1)), log_sigma_t1], 1)
            state = torch.cat([state, extra], 1)

        return state  # (n, state_dim)

    def simulate_step(self, spot, sigma):
        """Simulates new step based on the discretized dynamics from eq. (5)"""
        z = torch.randn_like(spot)
        spot_next = spot * torch.exp((-0.5 * sigma ** 2) * self.dt +
                                     sigma * torch.sqrt(self.dt) * z)
        return spot_next

    def reward(self, paths):
        # TODO: adjust this reward function to the different experiments (for example if we use bermudan options)
        #  the way it is right now, it is only useful to calibrate implied vol for black scholes model I think
        model_iv = self.opts.iv_calls(paths)  # (n_calls,)
        err = (model_iv - self.market_iv) ** 2
        return -err.sum()  # scalar

    def shaped_vanilla_reward_step(self, t, spot, vol_proxy):
        """
        Compute r̃_t  (equation ☆) for all vanilla calls.
        • spot         : (n,) tensor  S_t  for every path
        • vol_proxy    : (n,) tensor  σ_t  used in BS formula
        Returns scalar reward_t
        """
        # --- price each call on *each* path then average --------------
        iv_path = []

        for k, mat in self.opts.calls:
            tau = (mat - t) * self.dt
            tau = torch.full_like(spot, tau)  # broadcast
            # price each path with current σ_t
            price_i = utils.european_bs_price(spot, k, tau, vol_proxy)
            # average price -> implied vol (scalar per option)
            price_mean = price_i.mean()
            iv = utils.implied_vol_call(price_mean, self.S0, k, tau[0])  # scalar tensor
            iv_path.append(iv)

        # shape  (n_calls, n_paths)
        model_iv_t = torch.stack(iv_path)  # (M,n)

        # pre-stored market prices / IVs:
        market_iv = self.market_iv.to(self.device)  # (M,)

        err_t = (model_iv_t - market_iv).pow(2)  # (M,)
        E_t = err_t.sum()  # scalar

        # ---------- telescoping reward (fake reward approximation)
        reward_t = -(E_t - getattr(self, "_prev_E", torch.zeros(1, device=self.device)))
        self._prev_E = E_t.detach()  # store for next step

        return reward_t

    # TODO, ojo, segun chatHPT deberia introducir el tensor de market IV de la siguiente manera:
    # env.market_iv = torch.tensor([iv_mkt(k, m)  # <-- your data loader
    #                               for k, m in vanilla_grid],
    #                              device=env.dev)

