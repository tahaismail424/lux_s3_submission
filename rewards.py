import numpy as np

def calculate_rewards(weights_matrix, map_memory, enemy_memory, ally_memory, relic_points, match_result=None):
    """
    Calculate custom rewards for each ship based on the observtion space and predefined metrics.

    Args:
        map_memory (np.array): Map memory (W, H, 2) for recency and object IDs.
        enemy_memory (np.array): Enemy memory (N, 4) for enemy positions, energy, and recency.
        ally_memory (np.array): Ally memory (M, 4) for position, energy, and recency
        relic_poitns (np.array): Current relic points for both teams
        match_result (str): Match outcome ("win", "loss", "draw", or None for ongoing matches).

    Returns:
        rewards (list): List of rewards for each ship
    """
    # calculate reward components based on cur state (ship independent)
    exploration_reward = calculate_exploration_reward(map_memory, enemy_memory, ally_memory)
    attack_reward = calculate_attack_reward(enemy_memory)
    defense_reward = calculate_defense_reward(ally_memory)
    relic_reward = relic_points[0] - relic_points[1] # Difference in relic scores

    # match reward
    match_reward = 0
    if match_result == "win":
        match_reward = 100
    elif match_result == "loss":
        match_reward = -50

    rewards = []
    n_ships = weights_matrix.shape[0]
    for i in range(n_ships):
        # calculate reward for each ship
        weights = weights_matrix[i]
        total_reward = (
            weights[0] * exploration_reward +
            weights[1] * attack_reward + 
            weights[2] * defense_reward + 
            weights[3] * relic_reward +
            match_reward
        )

        rewards.append(total_reward)
    return np.array(rewards)

def calculate_exploration_reward(map_memory, enemy_memory, ally_memory):
    """
    Calculate the exploration reward based on map memory.
    Reward is higher for reducing unknown tiles and keeping recency scores low
    """
    recency_scores = np.sum(map_memory[:, :, 1]) + np.sum(enemy_memory[:, 2]) + np.sum(ally_memory[:, 2])  # recency scores
    unknown_tiles = np.sum(map_memory[:, :, 0] == -1) + np.sum(enemy_memory[:, 2] == -1) + np.sum(ally_memory[:, 2] == -1) # count of unknown tiles
    return -recency_scores - 5 * unknown_tiles # Scale unknown tile penalty
  
def calculate_attack_reward(enemy_memory):
    """
    Calculate the attack reward based on enemy memory.
    Reward is higher for reducing enemy energy and decommissioning ships.
    """
    enemy_energies = enemy_memory[:, 2] # energy values
    decommissioned_enemies = np.sum(enemy_energies == 0)
    return -np.sum(enemy_energies) + 5 * decommissioned_enemies

def calculate_defense_reward(ally_memory):
    """
    Calculate the defense reward based on ally states.
    Reward is higher for keeping aly energy high and preventing decommissioning.
    """
    ally_energies = ally_memory[:, 2] # energy values
    decommissioned_allies = np.sum(ally_energies == 0)
    return np.sum(ally_energies) - 5 * decommissioned_allies


    