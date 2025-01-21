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
        self.rnn = nn.GRU(input_size = 128 + 64 + 64 + 64 + 32 + 32 + 32, hidden_size=256, batch_first=True)

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

        # value head for action advantage
        self.value_head = nn.Linear(256 + 64, 1)
    
    def forward(
            self, map_memory, enemy_memory, 
            ally_memory, ship_state, 
            relic_points, match_points, 
            match_number, sap_range,
            hidden_state=None):
        # process map memory
        map_features = self.map_conv(map_memory)

        # process enemy states
        enemy_features = self.enemy_fc(enemy_memory.view(enemy_memory.size(0), -1))

        # process allied states
        ally_features = self.ally_fc(ally_memory.view(ally_memory.size(0), -1))

        # process indiviudal ship state
        ship_features = self.ship_fc(ship_state)

        # process relic points
        relic_features = self.relic_fc(relic_points)

        # process match points
        match_features = self.match_fc(match_points)

        # process match number
        match_num_features = self.match_num_fc(match_number.unsqueeze(-1))

        # combine feture
        combined_features = torch.cat([
            map_features, enemy_features, ally_features, 
            ship_features, relic_features, match_features, 
            match_num_features
        ], dim=-1)

        # pass through RNN for temporal memory
        rnn_out, hidden_state = self.rnn(combined_features.unsqueeze(1), hidden_state)
        rnn_out = rnn_out.squeeze(1)

        # combpute weight policy
        weights_out = self.weight_policy_head(rnn_out)

        # process weights vector
        weights_features = self.weights_fc(weights_out)

        # process sap range
        sap_range_features = self.sap_range_fc(sap_range)

        # compute action policy
        combined_action_input = torch.cat([rnn_out, weights_features, sap_range_features], dim=-1)
        action_probs = self.action_policy_head(combined_action_input)
        value = self.value_head(combined_action_input)

        return weights_out, action_probs, value, hidden_state
