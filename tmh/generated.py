
The network is composed of a convolutional neural network that takes in the board state as an input.
The output of the network is a probability distribution over all the possible moves.

The network is trained using a policy gradient method.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import math
import copy
import time
import os

from tqdm import tqdm
from collections import deque
from copy import deepcopy
from itertools import product
from functools import reduce

from utils import *

class PolicyNetwork(nn.Module):
    """
    A convolutional neural network that takes in the board state as an input and outputs a probability distribution
    over all the possible moves.
    """
    def __init__(self, input_size, output_size, hidden_size, num_filters, kernel_size):
        super(PolicyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, num_filters, kernel_size)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size)
        self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size)
        self.conv4 = nn.Conv2d(num_filters, num_filters, kernel_size)
        self.conv5 = nn.Conv2d(num_filters, num_filters, kernel_size)
        self.fc1 = nn.Linear(num_filters * (input_size - 4 * (kernel_size - 1)) * (input_size - 4 * (kernel_size - 1)), hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class PolicyGradientAgent():
    """
    A reinforcement learning agent that learns how to play tic-tac-toe using policy gradient methods.
    """
    def __init__(self, input_size, output_size, hidden_size, num_filters, kernel_size, learning_rate, gamma, decay_rate, decay_freq, cuda):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.decay_freq = decay_freq
        self.cuda = cuda

        self.policy_network = PolicyNetwork(input_size, output_size, hidden_size, num_filters, kernel_size)
        if self.cuda:
            self.policy_network.cuda()
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=decay_freq, gamma=decay_rate)

        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []

    def reset_episode(self):
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []

    def select_action(self, state, legal_moves):
        """
        Selects an action to take based on the policy network and the current state.
        """
        state = torch.from_numpy(state).float().unsqueeze(0)
        if self.cuda:
            state = state.cuda()
        probs = self.policy_network(state)
        probs = probs * legal_moves
        probs = probs / probs.sum()
        action = np.random.choice(self.output_size, p=probs.cpu().detach().numpy())
        return action

    def update_policy(self):
        """
        Updates the policy network using the states, actions, and rewards of the current episode.
        """
        R = 0
        rewards = []
        for r in self.episode_rewards[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards)
        if self.cuda:
            rewards = rewards.cuda()
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
        for (log_prob, value), r in zip(self.episode_actions, rewards):
            reward = r - value.item()
            policy_losses.append(-log_prob * reward)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_losses).sum()
        policy_loss.backward()
        self.optimizer.step()
        del self.episode_states[:]
        del self.episode_actions[:]
        del self.episode_rewards[:]

    def save_model(self, path):
        """
        Saves the model to the specified path.
        """
        torch.save(self.policy_network.state_dict(), path)

    def load_model(