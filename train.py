import torch
import torch.optim as optim
from luxai_s3.wrappers import LuxAIS3GymEnv

def train_agent(agent, num_episodes, gamma=0.99, lambda_=0.95, entropy_coeff=0.01, value_coeff=0.5):
    """
    Training loop for Lux Agent using an actor-critic methodology
    """
    env = LuxAIS3GymEnv(numpy_output=True)
    obs, info = env.reset()

    env_cfg = A