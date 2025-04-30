import torch, torch.nn as nn, torch.optim as optim
from torch.distributions.normal import Normal
from environment import MARLVolEnv
import math
import numpy as np
import os, pathlib


EPS = 1e-8


def train(s0, env: MARLVolEnv, policy, critic, n_epochs=3000, gamma=0.98, lr=1e-4, clip_grad=5.,
          save_every=100, save_dir="checkpoints"):
    """
    :param save_dir:
    :param save_every:
    :param s0: current spot price
    :param env: MARLVolEnv may paths simulator
    :param policy: shared policy (actor) network
    :param critic: shared critic (value) network
    :param n_epochs: number of epochs
    :param gamma: discount factor used in the advantage calculation (it has nothing to do with interest rate!)
    :param lr: optimiser hyperparameter
    :param clip_grad: optimiser hyperparameter
    :return: None
    """
    pathlib.Path(save_dir).mkdir(exist_ok=True)
    opt_p = optim.Adam(policy.parameters(),  lr=lr)
    opt_v = optim.Adam(critic.parameters(), lr=lr)

    for epoch in range(n_epochs):

        env.reset()

        # temporary Python lists that will accumulate trajectory data needed for the A2C losses
        logp_list, value_list, reward_list, state_list, advantage_list = [], [], [], [], []

        # init paths
        spot = torch.full((env.n_agents,), s0, device=env.device)
        # Initialize the volatility at 0.1
        log_sigma = torch.zeros(env.n_agents, device=env.device) + math.log(0.1)
        spot_hist, logsig_hist = [spot], [log_sigma]

        # per-time-step simulation ---------------------------------------
        for t in range(env.T):
            # state vectors (should be modified if I want to include bermudean options in the future)
            state = env.build_state(t, spot, log_sigma)

            # forward pass. We sample an action distribution from the policy (actor) network
            # in the form of a mean and variance that correspond to a gaussian distribution
            mu, log_std = policy(state)
            std = log_std.exp()  # remember, this is the standard deviation of the distribution of the log-std of the
            # diffusion process (what we later call log_sigma_act)

            # Basis-player exploration. choose n-basis basis players uniformly at random among the n paths.
            # For this we randomly choose n_basis paths sampling numbers from 0 to n (total number of paths)
            basis_idx = torch.randint(0, env.n_agents, (env.n_basis,),
                                      device=env.device)
            # sample noise for each of the basis players (this is the Z_hat in eq. (15))
            noise_basis = torch.randn(env.n_basis, device=env.device)

            # Interpolation of all players using basis players
            # work under no-grad to avoid polluting autograd graph

            with torch.no_grad():
                # 1. build features for the basis players
                if not torch.isfinite(torch.as_tensor(spot, dtype=torch.float32)).all():
                    bad = spot[~torch.isfinite(spot)]
                    print("Non-finite spot values:", bad[:10])

                basis_features = torch.log(torch.clamp(spot[basis_idx], min=EPS)).unsqueeze(1) \
                    .detach().cpu().numpy()

                if not torch.isfinite(torch.as_tensor(basis_features, dtype=torch.float32)).all():
                    idx = ~np.isfinite(basis_features.squeeze())
                    print("log(spot) that blew up:", spot[basis_idx][idx])

                # 2. same for the target noise
                noise_np = noise_basis.detach().cpu().numpy()

                # 3. fit k-NN   (shape:  np × 1  →  np × 1)
                env.knn.fit(basis_features, noise_np)

                # 4. interpolate to *all* paths
                all_features = torch.log(torch.clamp(spot, min=EPS)).unsqueeze(1) \
                    .detach().cpu().numpy()  # (n, 1)
                noise_all = torch.tensor(env.knn.predict(all_features),
                                         device=env.device)

            # Sample action for every player
            # action = mu + noise * std      (eq. 15)
            log_sigma_act = mu + noise_all * std
            sigma_act = log_sigma_act.exp()

            # Store log probability of sampled log_sigma_act
            dist = Normal(mu, std)
            logp = dist.log_prob(log_sigma_act)

            state_list.append(state)
            logp_list.append(logp)

            # Sample and store the value prediction of the critic network and store it
            value_t = critic(state).detach()
            value_list.append(value_t)

            # Simulate next step (diffuse the process according to the discretized dynamics from eq. (5))
            spot_next = env.simulate_step(spot, sigma_act)
            spot = spot_next

            # calculate fake reward approximation to avoid sparcity in the reward
            r_t = env.shaped_vanilla_reward_step(t, spot, sigma_act.detach())
            # broadcast same team reward to every path
            reward_list.append(r_t.expand(env.n_agents))

            # save the sampled log sigma for the new state
            log_sigma = log_sigma_act
            # save spot and logsig history
            spot_hist.append(spot)
            logsig_hist.append(log_sigma)

        # stack trajectories
        spot_paths = torch.stack(spot_hist)  # (T+1,n)
        rewards = torch.stack(reward_list)  # (T,n)
        values = torch.stack(value_list)  # (T,n)
        returns = torch.zeros_like(values)
        adv = torch.zeros_like(values)

        #  advantage & return computations
        G = torch.zeros(env.n_agents, device=env.device)
        for t in reversed(range(env.T)):
            G = rewards[t] + gamma * G
            returns[t] = G
            adv[t] = G - values[t]
        # flatten

        adv_f = adv.flatten().detach()
        ret_f = returns.flatten().detach()
        logp_f = torch.stack(logp_list).flatten()
        states_f = torch.cat(state_list)

        # ----- critic update -------------------------------------------
        opt_v.zero_grad()
        value_pred = critic(states_f.detach())
        value_loss = nn.functional.mse_loss(value_pred, ret_f)
        value_loss.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), clip_grad)
        opt_v.step()

        # ----- actor update (A2C) --------------------------------------
        opt_p.zero_grad()
        pol_loss = -(logp_f * adv_f).mean()
        pol_loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), clip_grad)
        opt_p.step()

        if epoch % save_every == 0 or epoch == n_epochs:
            torch.save(policy.state_dict(),
                       f"{save_dir}/policy_{epoch:06d}.pt")
            torch.save(critic.state_dict(),
                       f"{save_dir}/critic_{epoch:06d}.pt")
            # optional: keep “latest” symlinks
            torch.save(policy.state_dict(), f"{save_dir}/policy_latest.pt")
            torch.save(critic.state_dict(), f"{save_dir}/critic_latest.pt")

        # Print accumulated reward for the episode
        R = rewards.sum()
        if (epoch+1) % 5 == 0:
            print(f"Iter {epoch+1:4d} | R = {R.item():.3e} "
                  f"| π loss {pol_loss.item():.3e} | V loss {value_loss.item():.3e}")
            # substitute the value below for the mean of the sigma distribution?
            print(f'Average iv of agents in last period: {torch.mean(torch.exp(logsig_hist[-1]))}')