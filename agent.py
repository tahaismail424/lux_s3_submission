import numpy as np
class Agent():
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        self.env_cfg = env_cfg
        self.obs_history = []

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

        self.enemy_memory = -np.ones((env_cfg["max_units"], 4))
        self.allied_memory = -np.ones((env_cfg["max_units"], 4))
        START_ENERGY = 100
        self.enemy_memory[:, 2] = START_ENERGY
        self.allied_memory[:, 2] = START_ENERGY

        # safe knowledge of relic location
        self.relic_memory = None


    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """implement this function to decide what actions to send to each available unit. 
        
        step is the current timestep number of the game starting from 0 going up to max_steps_in_match * match_count_per_episode - 1.
        """
        features = self.process_obs(obs)
        unit_mask = np.array(obs["units_mask"][self.team_id]) # shape (max_units, )
        unit_positions = np.array(obs["units"]["position"][self.team_id]) # shape (max_units, 2)
        unit_energys = np.array(obs["units"]["energy"][self.team_id]) # shape (max_units, 1)
        observed_relic_node_positions = np.array(obs["relic_nodes"]) # shape (max_relic_nodes, 2)
        observed_relic_nodes_mask = np.array(obs["relic_nodes_mask"]) # shape (max_relic_nodes, )
        team_points = np.array(obs["team_points"]) # points of each team, team_points[self.team_id] is the points of the your team

        return actions
    
    def process_obs(self, obs, match_number):
        """
        This functions processes observation space 
        into feature vectors to input into the agent learning network
        """

        # update map memory for all true values in sensor mask
        map_memory = self.map_memory
        vis = obs["sensor_mask"]
        map_memory[vis][0] = obs["map_features"]["tile_type"][vis]
        map_memory[vis][1] = obs["map_features"]["energy"][vis]
        map_memory[vis][2] = 0

        # update map memory with relic positions
        if self.relic_memory == None:
            self.relic_memory = np.zeros((obs["relic_nodes_mask"].shape[0], 2))
        self.relic_memory[obs["relic_nodes_mask"]] = obs["relic_nodes"][obs["relic_nodes_mask"]]
        map_memory[self.relic_memory[:, 0], self.relic_memory[:, 1]][0] = 3
        map_memory[self.relic_memory[:, 0], self.relic_memory[:, 1]][2] = 0

        # increment recency score for all discovered positions
        map_memory[map_memory[:, :, 2] != -1] += 1
        self.map_memory = map_memory

        # update enemy memory
        enemy_memory = self.enemy_memory
        vis = obs[self.opp_team_id]["units_mask"]
        enemy_memory[vis][0] = obs["units"]["position"][self.opp_team_id][vis][0]
        enemy_memory[vis][1] = obs["units"]["position"][self.opp_team_id][vis][1]
        enemy_memory[vis][2] = obs["units"]["energy"][self.opp_team_id][vis][0]
        enemy_memory[vis][3] = 0
        
        # increment recency score for all discovered units
        enemy_memory[enemy_memory[:, 3] != -1] += 1
        self.enemy_memory = enemy_memory

        # update ally memory
        ally_memory = self.allied_memory
        vis = obs[self.team_id]["units_mask"]
        ally_memory[vis][0] = obs["units"]["position"][self.team_id][vis][0]
        ally_memory[vis][1] = obs["units"]["position"][self.team_id][vis][1]
        ally_memory[vis][2] = obs["units"]["energy"][self.team_id][vis][0]
        ally_memory[vis][3] = 0
        
        # increment recency score for all discovered units
        ally_memory[ally_memory[:, 3] != -1] += 1
        self.allied_memory = ally_memory

        # get relic points for both teams
        relic_points = obs["team_points"]

        # get match points for both teams
        match_points = obs["team_wins"]

       # get sap range
        sap_range = obs["info"]["env_cfg"]["unit_sap_range"]

        obs_processed = (
            map_memory,
            enemy_memory,
            ally_memory,
            relic_points,
            match_points, 
            match_number,
            sap_range
        )
        self.obs_history.append(obs_processed)
        return obs_processed

