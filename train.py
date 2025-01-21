import torch
import torch.optim as optim

def train_agent(agent, num_episodes, gamma=0.99, lambda_=0.95, entropy_coeff=0.01, value_coeff=0.5):
    """
    Training loop for Lux Agent using an actor-critic methodology
    """