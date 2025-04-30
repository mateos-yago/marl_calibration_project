import torch, math
from policy_network import SharedPolicy


def get_results():
    policy = SharedPolicy(state_dim=3)
    policy.load_state_dict(torch.load('checkpoints/policy_latest.pt'))
    policy.eval()

    # --- 2.  build the state you care about -----------------------
    S0 = 1.0  # same as in training
    log_S0 = math.log(S0)
    t_norm = torch.tensor([0.0])  # start of the path
    init_log_sigma = torch.tensor([math.log(0.22)])  # any guess, not critical

    state = torch.stack([t_norm, torch.tensor([log_S0]), init_log_sigma], 1)

    # --- 3.  forward pass -----------------------------------------
    with torch.no_grad():
        mu, _ = policy(state)  # ignore log_std for deterministic vol
    calibrated_vol = mu.exp().item()

    print(f"Calibrated σ : {calibrated_vol:.4f}")
