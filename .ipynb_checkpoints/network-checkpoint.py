import torch
import torch.nn as nn

class AgentNetwork(nn.Module):
    def __init__(self, map_dims, num_ships, action_space):
        super(AgentNetwork, self).__init__()

        # Map memory feature extractor (ConvNet)
        self.map_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(map_dims[0] * map_dims[1] * 32, 128)
        )

        # Enemy state feature extractor (FCN)
        self.enemy_fc = nn.Sequential(
            nn.Linear(num_ships * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        # Allied state feature extractor (FCN)
        self.ally_fc = nn.Sequential(
            nn.Linear(num_ships * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        # Ship state feature extractor (FCN)
        self.ship_fc = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU()
        )

        # Relic points feature extractor (FCN)
        self.relic_fc = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU()
        )

        # Current match standings (FCN)
        self.match_fc = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU()
        )

        # Match number encoding (FCN with temporal encoding)
        self.match_num_fc = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )

        # weights vector feature extractor (FCN)
        self.weights_fc = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU()
        )

        # RNN for temporal memory
        self.rnn = nn.GRU(input_size = 128 + 64 + 64 + 64 + 32 + 32, hidden_size=256, batch_first=True)

        # Weight policy head
        self.weight_policy_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 4), # output weights for exploration, attack, defense, relic
            nn.Softmax(dim=-1)
        )

        # sap range encoding for action selection
        self.sap_range_fc = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU()
        )

        # action policy head
        self.action_policy_head = nn.Sequential(
            nn.Linear(256 + 64 + 32, 128), # hidden state + weights
            nn.ReLU(),
            nn.Linear(128, action_space)
        )

        # sap offset
        self.offset_head = nn.Sequential(
            nn.Linear(256 + 64 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        ) # output: dx and dy

        # value head for action advantage
        self.value_head = nn.Linear(256 + 64, 1)
    
    def forward(self, shared_inputs, ship_states, hidden_state=None):
        # unravel shared_inputs dict
        map_memory = shared_inputs["map_memory"] 
        enemy_memory = shared_inputs["enemy_memory"]
        ally_memory = shared_inputs["ally_memory"]
        relic_points = shared_inputs["relic_points"]
        match_points = shared_inputs["match_points"]
        sap_range = shared_inputs["sap_range"]

        # process map memory
        map_features = self.map_conv(map_memory)

        # process enemy states
        enemy_features = self.enemy_fc(enemy_memory.view(enemy_memory.size(0), -1))

        # process allied states
        ally_features = self.ally_fc(ally_memory.view(ally_memory.size(0), -1))

        # process relic points
        relic_features = self.relic_fc(relic_points)

        # process match points
        match_features = self.match_fc(match_points)

        # process individual ship state
        ship_features = self.ship_fc(ship_states)

        # shared embedding
        shared_embedding = torch.cat([
            map_features, enemy_features, ally_features,
            relic_features, match_features
        ], dim=-1)

        # expand shared embedding to match number of ships
        shared_embedding = shared_embedding.unsqueeze(1).repeat(1, ship_states.size(0), 1) # (batch_size, num_ships, embedding_dim)


        # combine feture
        combined_features = torch.cat([shared_embedding, ship_features], dim=-1) # (batch_size, embedding_dim + ship_embedding_dim)

        # pass through RNN for temporal memory
        rnn_out, hidden_state = self.rnn(combined_features, hidden_state)

        # combpute weight policy
        weights_out = self.weight_policy_head(rnn_out)

        # process weights vector
        weights_features = self.weights_fc(weights_out)

        # process sap range
        sap_range_features = self.sap_range_fc(sap_range).unsqueeze(1).repeat(1, ship_states.size(0), 1)

        # compute action policy
        combined_action_input = torch.cat([rnn_out, weights_features, sap_range_features], dim=-1)
        action_probs = self.action_policy_head(combined_action_input)
        sap_offset = self.offset_head(combined_action_input)
        value = self.value_head(combined_action_input)

        return weights_out, action_probs, sap_offset, value, hidden_state

def compute_network_difference(agent1, agent2):
    """
    Compute the L2 norm of the difference between two networks.

    Args:
        agent1: The current network (Player 1).
        agent2: The lagging network (Player 2).

    Returns:
        float: L2 norm of the parameter differences.
    """
    difference = 0.0
    for p1, p2 in zip(agent1.parameters(), agent2.parameters()):
        difference += torch.norm(p1 - p2).item() ** 2
    return difference ** 0.5

def has_converged(win_rates, network_differences, win_rate_threshold=0.7, win_rate_range=0.05, diff_threshold=0.01, k=10):
    """
    Check if training has converged based on win rates and network differences.

    Args:
        win_rates (list): Rolling win rates for Player 1.
        network_differences (list): Rolling network differences.
        win_rate_threshold (float): Minimum average win rate for Player 1.
        win_rate_range (float): Allowed oscillation range for win rates.
        diff_threshold (float): Threshold for network difference stabilization.
        k (int): Number of recent episodes to check for stabilization.

    Returns:
        bool: True if convergence criteria are met.
    """
    # Check win rate stability
    recent_win_rates = win_rates[-k:]
    if len(recent_win_rates) < k:
        return False
    if max(recent_win_rates) - min(recent_win_rates) > win_rate_range:
        return False
    if sum(recent_win_rates) / len(recent_win_rates) < win_rate_threshold:
        return False

    # Check network difference stabilization
    recent_differences = network_differences[-k:]
    if len(recent_differences) < k:
        return False
    if max(recent_differences) > diff_threshold:
        return False

    return True