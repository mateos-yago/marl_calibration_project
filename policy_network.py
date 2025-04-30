import torch.nn as nn


class SharedPolicy(nn.Module):
    """Outputs the parameters of a probability distribution for the volatility.
    We assume the volatility distribution to be gaussian"""

    def __init__(self, state_dim, hidden=64):
        super().__init__()
        # ── Feature extractor (the “backbone”) ────────────────────────────────────
        # Takes a state_dim-dimensional input (state of the MDP) and maps it to a hidden-dimensional
        # feature vector via two fully‐connected layers with Tanh activations.
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        # ── Policy heads ─────────────────────────────────────────────────────────
        # Each head takes the shared hidden features and produces a scalar output:
        # • mu_head      outputs the mean μ(s) of the Gaussian policy
        # • logstd_head  outputs log σ(s), i.e. the log of the standard deviation
        # EXPLANATION: We want the policy to sample the log of the volatility. The action a_t in the paper
        #  is actually the log of the standard deviation that is used in the diffusion process
        self.mu_head = nn.Linear(hidden, 1)  # mean of log(σ)
        self.logstd_head = nn.Linear(hidden, 1)  # log std of the standard deviation distribution of the
        # action, which is itself the log of std of the diffusion process (log_sigma_act

    def forward(self, state):
        # 1) Extract features
        x = self.backbone(state)  # → (batch_size, hidden)

        # 2) Compute raw outputs
        mu = self.mu_head(x)  # → (batch_size, 1)
        log_std = self.logstd_head(x)  # → (batch_size, 1)

        # 3) Clamp log_std and mu to keep in a reasonable range
        log_std = log_std.clamp(-5, 3)
        mu = mu.clamp(-3, 3)

        # 4) Remove the trailing singleton dimension
        return mu.squeeze(-1), log_std.squeeze(-1)
