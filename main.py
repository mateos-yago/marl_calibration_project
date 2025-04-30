from options_set import OptionSet
from environment import MARLVolEnv
import torch
from policy_network import SharedPolicy
from critic_network import CriticNetwork
from train import train
from results import get_results
import utils
import numpy as np


def main():
    S0, K, iv_mkt, days = 1.0, 1.0, 0.22, 20
    tau = days/252
    r = 0.02

    # price_mkt = utils.european_bs_price(S0, K, tau, iv_mkt, r=r)

    vanilla = [(K, days)]

    opt_set = OptionSet(vanilla, r)  # no Bermudan yet

    n_paths = 2000  # small, should change it when I try to calibrate more complex models
    T_steps = days  # simulate daily
    env = MARLVolEnv(n_paths, S0, T_steps, 1 / 252, opt_set, iv_mkt,
                     n_basis=20, device='cpu')

    obs_dim = 3  # (t_norm, logS, logσ)
    policy = SharedPolicy(obs_dim).cpu()
    critic = CriticNetwork(obs_dim).cpu()
    train(S0, env, policy, critic, n_epochs=1000)

    print('\\ Training finished')


if __name__ == '__main__':
    #main()
    get_results()
