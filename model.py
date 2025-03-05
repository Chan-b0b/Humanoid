import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import gymnasium as gym
import torch.nn.init as init

class CustomLegPolicy(nn.Module):
    """Custom policy network that only controls the legs, keeping arm actions at zero."""
    
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(CustomLegPolicy, self).__init__()

        # Simple 2-layer MLP
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # Outputs actions for all joints
        self.activation = nn.ELU()
        self.last_activation = nn.Tanh()
        self._initialize_weights()

    def forward(self, obs):
        x = self.activation(self.fc1(obs))
        actions = self.last_activation(self.fc2(x))

        # # Assuming first 6 outputs control legs, next ones are arms
        # leg_actions = actions[:, :11]  # First 6 control legs
        # arm_actions = torch.zeros_like(actions[:, 11:])  # Arms always zero

        # return torch.cat([leg_actions, arm_actions], dim=1)  # Merge outputs
        return actions

    def _initialize_weights(self):
        """Initialize the last layer to output values near zero."""
        init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')  # Good for hidden layers
        init.zeros_(self.fc1.bias)  # Set bias to zero

        init.normal_(self.fc2.weight, mean=0.0, std=0.01)  # Small weight std
        init.zeros_(self.fc2.bias)  # Zero bias ensures near-zero outputs

class CNNExtractor(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for humanoid policy.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CNNExtractor, self).__init__(observation_space, features_dim)

        num_joints = 100  # Assume humanoid has 29 joints
        input_channels = 15 # qpos and qvel

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()  # Flatten to pass to linear layers
        )

        # Compute CNN output shape dynamically
        with torch.no_grad():
            dummy_input = torch.randn(1, input_channels, num_joints)  # (batch, channels, num_joints)
            cnn_output_dim = self.cnn(dummy_input).shape[1]

        # Final linear layer to map CNN output to desired feature dimension
        self.fc = nn.Sequential(
            nn.Linear(cnn_output_dim, features_dim),
            nn.ReLU()
        )

        # self.fc_policy = nn.Sequential(
        #     nn.Linear(features_dim, features_dim),
        #     nn.ReLU()
        # )
        # self.fc_value = nn.Sequential(
        #     nn.Linear(features_dim, 64),
        #     nn.ReLU()
        # )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN feature extractor.
        """
        # Split qpos and qvel
        obs_reshaped = observations.view(observations.shape[0], 15, -1)

        # Pass through CNN
        features = self.cnn(obs_reshaped)
        
        features = self.fc(features)
        return features

def normal_init(m):
    if isinstance(m, nn.Linear):  
        torch.nn.init.normal_(m.weight, mean=0.0, std=1.0)  # Standard Normal
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

class CustomPolicy(ActorCriticPolicy):
    """PPO Policy using the Custom Leg Controller"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Replace actor network with our custom one
        self.action_net = CustomLegPolicy(
            input_dim=64, 
            output_dim=self.action_space.shape[0]
        )
        
        # self.apply(normal_init)  # Apply the initialization

        with torch.no_grad():
            self.log_std.data.fill_(-2) 
