import numpy as np

def compute_advantages(rewards, values, gamma=0.99, lambda_=0.95):
    """
    Compute discounted returns and advantages for actor-critic training.

    Args:
        rewards (list): List of rewards at each time step.
        values (list): List of state values (from the value head) at each time step.
        gamma (float): Discount factor for rewards.
        lambda_ (float): GAE (Generalized Advantage Estimation) smoothing parameter.

    Returns:
        returns (np.array): Discounted returns for each time step.
        advantages (np.array): Advantages for each time step.
    """
    # initialize arrays
    returns = np.zeros_like(rewards, dtype=np.float32)
    advantages = np.zeros_like(rewards, dtype=np.float32)

    # Compute returns and advantages
    next_return = np.zeros_like(rewards[0], dtype=np.float32)
    next_value = np.zeros_like(rewards[0], dtype=np.float32)
    for t in reversed(range(len(rewards))):
        # Discounted return
        returns[t] = rewards[t] + gamma * next_return
        next_return = returns[t]

        # temporal difference and advantage
        td_error = rewards[t] + gamma * next_value - values[t]
        advantages[t] = td_error + gamma * lambda_ * advantages[t + 1] if t + 1 < len(rewards) else td_error
        next_value = values[t]
    
    return returns, advantages


def compute_weight_loss(log_probs, advantages, values, returns, entropy_coeff=0.01, value_coeff=0.5):
    """
    Compute the loss for the weight policy head.

    Args:
        log_probs (torch.Tensor): Log probabilities of the weights
        advantages (torch.Tensor): Computed advantages for each step.
        values (torch.Tensor): Predicted values from the value head.
        returns (torch.Tensor): Discounted returns for each step.
        entropy_coeff (float): Coefficient for entropy regularization.
        value_coeff (float): Coefficient for value loss.

    Returns:
        torch.Tensor: Total loss for the weight policy head.
    """
    # Policy loss
    policy_loss = -(log_probs * advantages.detach().unsqueeze(-1)).sum(dim=-1).mean()

    # Value loss
    value_loss = value_coeff * (returns - values).pow(2).mean()

    # Entropy loss
    entropy_loss = -entropy_coeff * (log_probs.exp() * log_probs).sum(-1).mean()

    return policy_loss + value_loss + entropy_loss

def compute_action_loss(log_probs, advantages, values, returns, entropy_coeff=0.01, value_coeff=0.5):
    """
    Compute the loss for the weight policy head.

    Args:
        log_probs (torch.Tensor): Log probabilities of the weights
        advantages (torch.Tensor): Computed advantages for each step.
        values (torch.Tensor): Predicted values from the value head.
        returns (torch.Tensor): Discounted returns for each step.
        entropy_coeff (float): Coefficient for entropy regularization.
        value_coeff (float): Coefficient for value loss.

    Returns:
        torch.Tensor: Total loss for the weight policy head.
    """
    # Policy loss
    policy_loss = -(log_probs * advantages.detach().unsqueeze(-1)).sum(dim=-1).mean()

    # Value loss
    value_loss = value_coeff * (returns - values).pow(2).mean()

    # Entropy loss
    entropy_loss = -entropy_coeff * (log_probs.exp() * log_probs).sum(-1).mean()

    return policy_loss + value_loss + entropy_loss
            