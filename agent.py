import numpy as np
from network import AgentNetwork
import torch

class Agent():
    def __init__(self, player: str, env_cfg, net: AgentNetwork, device: torch.device) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        self.env_cfg = env_cfg
        self.net = net
        self.device = device

        # save map information in agent 
        self.relic_node_positions = []
        self.discovered_relic_nodes_ids = set()
        self.unit_explore_locations = dict()

        # get map dimensions
        self.map_width = env_cfg["map_width"]
        self.map_height = env_cfg["map_height"]
        
        # save map information in agent
        self.map_memory = np.zeros((self.map_width, self.map_height, 3))
        self.map_memory[:, :, 0] = -1
        self.map_memory[:, :, 2] = -1

        self.enemy_memory = -np.ones((env_cfg["max_units"], 4), dtype=int)
        self.allied_memory = -np.ones((env_cfg["max_units"], 4), dtype=int)
        START_ENERGY = 100
        self.enemy_memory[:, 2] = START_ENERGY
        self.allied_memory[:, 2] = START_ENERGY

        # save knowledge of relic location
        self.relic_memory = None

    def act_train(self, step: int, obs, remainingOverageTime: int = 60):
        """implement this function to calculate network outputs for agent training.
        """
        (map_memory, enemy_memory, ally_memory, 
         relic_points, match_points, 
         sap_range) = self.process_obs(obs)
        
        shared_features = {
            "map_memory": torch.tensor(map_memory, dtype=torch.float32).to(self.device),
            "enemy_memory": torch.tensor(enemy_memory, dtype=torch.float32).to(self.device),
            "ally_memory": torch.tensor(ally_memory, dtype=torch.float32).to(self.device),
            "relic_points": torch.tensor(relic_points, dtype=torch.float32).to(self.device),
            "match_points": torch.tensor(match_points, dtype=torch.float32).to(self.device),
            "sap_range": torch.tensor(sap_range, dtype=torch.float32).to(self.device)
        }

        ship_states = torch.tensor(self.allied_memory, dtype=torch.float32).to(self.device)
        return self.net(shared_features, ship_states)


    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """implement this function to decide what actions to send to each available unit. 
        
        step is the current timestep number of the game starting from 0 going up to max_steps_in_match * match_count_per_episode - 1.
        """
        (map_memory, enemy_memory, ally_memory, 
         relic_points, match_points,
         sap_range) = self.process_obs(obs)
        
        shared_features = {
            "map_memory": torch.tensor(map_memory, dtype=torch.float32).to(self.device),
            "enemy_memory": torch.tensor(enemy_memory, dtype=torch.float32).to(self.device),
            "ally_memory": torch.tensor(ally_memory, dtype=torch.float32).to(self.device),
            "relic_points": torch.tensor(relic_points, dtype=torch.long).to(self.device),
            "match_points": torch.tensor(match_points, dtype=torch.long).to(self.device),
            "sap_range": torch.tensor(sap_range, dtype=torch.float32).to(self.device)
        }

        ship_states = torch.tensor(self.allied_memory, dtype=torch.float32).to(self.device)
        _, action_probs, sap_offset, _, _ = self.net(shared_features, ship_states)
        actions = self.sample_actions(action_probs, sap_offset)
        return actions
    
    def process_obs(self, obs):
        """
        This functions processes observation space 
        into feature vectors to input into the agent learning network
        """

        # update map memory for all true values in sensor mask
        map_memory = self.map_memory
        vis = obs["sensor_mask"]
        if map_memory[vis].size != 0:
            map_memory[vis, 0] = obs["map_features"]["tile_type"][vis]
            map_memory[vis, 1] = obs["map_features"]["energy"][vis]
            map_memory[vis, 2] = 0

        # update map memory with relic positions
        if not isinstance(self.relic_memory, np.ndarray):
            self.relic_memory = -np.ones((obs["relic_nodes_mask"].shape[0], 2), dtype=int)
        self.relic_memory[obs["relic_nodes_mask"]] = obs["relic_nodes"][obs["relic_nodes_mask"]]
        if self.relic_memory[self.relic_memory[:, 0] != -1].size != 0 and self.relic_memory[self.relic_memory[:, 1] != -1].size != 0:
            map_memory[self.relic_memory[self.relic_memory[:, 0] != -1][:, 0], self.relic_memory[self.relic_memory[:, 1] != -1][:, 1], 0] = 3
            map_memory[self.relic_memory[self.relic_memory[:, 0] != -1][:, 0], self.relic_memory[self.relic_memory[:, 1] != -1][:, 1], 2] = 0

        # increment recency score for all discovered positions
        map_memory[map_memory[:, :, 2] != -1, 2] += 1
        self.map_memory = map_memory

        # update enemy memory
        enemy_memory = self.enemy_memory
        vis = obs["units_mask"][self.opp_team_id]
        if enemy_memory[vis].size != 0:
            enemy_memory[vis, 0] = obs["units"]["position"][self.opp_team_id][vis][:, 0]
            enemy_memory[vis, 1] = obs["units"]["position"][self.opp_team_id][vis][:, 1]
            enemy_memory[vis, 2] = obs["units"]["energy"][self.opp_team_id][vis]
            enemy_memory[vis, 3] = 0
        
        # increment recency score for all discovered units
        enemy_memory[enemy_memory[:, 3] != -1, 3] += 1
        self.enemy_memory = enemy_memory

        # update ally memory
        ally_memory = self.allied_memory
        vis = obs["units_mask"][self.team_id]
        if ally_memory[vis].size != 0:
            ally_memory[vis, 0] = obs["units"]["position"][self.team_id][vis][:, 0]
            ally_memory[vis, 1] = obs["units"]["position"][self.team_id][vis][:, 1]
            ally_memory[vis, 2] = obs["units"]["energy"][self.team_id][vis]
            ally_memory[vis, 3] = 0
        
        # increment recency score for all discovered units
        ally_memory[ally_memory[:, 3] != -1, 3] += 1
        self.allied_memory = ally_memory

        # get relic points for both teams
        relic_points = obs["team_points"]

        # get match points for both teams
        match_points = obs["team_wins"]

       # get sap range
        sap_range = self.env_cfg["unit_sap_range"]

        obs_processed = (
            map_memory,
            enemy_memory,
            ally_memory,
            relic_points,
            match_points, 
            sap_range
        )
        return obs_processed
    
    def sample_actions(self, action_probs, sap_offset):
        """
        Sample actions for a batch of ships based on action outputs.

        Args:
            action_outputs (dict): Output of the action policy head containing:
                                - "action_probs": (batch_size, action_space)
                                - "dx": (batch_size, 1)
                                - "dy": (batch_size, 1)

        Returns:
            torch.Tensor: A tensor of shape (batch_size, 3) where each row is:
                        - action index
                        - dx (if sap, otherwise 0)
                        - dy (if sap, otherwise 0)
        """
        
        sap_range = self.env_cfg["unit_sap_range"] 
        dx = torch.round(sap_offset.squeeze(0)[:, 0]) * (sap_range * 2) - sap_range                     # (batch_size, 1) - scaled
        dy = torch.round(sap_offset.squeeze(0)[:, 1]) * (sap_range * 2) - sap_range             # (batch_size, 1) - scalled
        
        # Sample the discrete action
        _, batch_size, action_space = action_probs.shape
        action_indices = torch.multinomial(action_probs.squeeze(0), 1).squeeze(-1)  # (batch_size,)

    
        # Initialize output dx and dy
        sampled_dx = torch.zeros(batch_size, device=self.device)
        sampled_dy = torch.zeros(batch_size, device=self.device)
        
        # Apply dx, dy only for sap actions
        sap_action_indices = (action_indices == action_space - 1)  # Assuming sap is the last action
        sampled_dx[sap_action_indices] = dx[sap_action_indices].squeeze(-1)
        sampled_dy[sap_action_indices] = dy[sap_action_indices].squeeze(-1)
        
        # Combine into final output (batch_size, 3)
        actions = torch.stack([action_indices.float(), sampled_dx, sampled_dy], dim=-1)
        return actions.detach().cpu().numpy().astype(int)

