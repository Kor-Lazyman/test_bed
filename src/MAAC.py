import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random


class AttentionCritic(nn.Module):
    """
    Attention-based critic network that evaluates actions taken by all agents.
    Uses scaled dot-product attention to model agent interactions.

    Args:
        obs_dim (int): Dimension of local observation space
        action_dim (int): Dimension of action space
        n_agents (int): Number of agents
        hidden_dim (int): Size of hidden layers
    """

    def __init__(self, obs_dim, action_dim, n_agents, hidden_dim=64):
        super(AttentionCritic, self).__init__()

        # Encoder network: processes concatenated observation and action
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Networks for attention mechanism
        # Generates keys for attention
        self.key_net = nn.Linear(hidden_dim, hidden_dim)
        # Generates queries for attention
        self.query_net = nn.Linear(hidden_dim, hidden_dim)
        # Generates values for attention
        self.value_net = nn.Linear(hidden_dim, hidden_dim)

        # Final network to produce Q-values
        self.final_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.n_agents = n_agents
        self.hidden_dim = hidden_dim

    def forward(self, obs, actions):
        """
        Forward pass of the critic network.

        Args:
            obs: Agent observations [batch_size, n_agents, obs_dim]
            actions: Agent actions [batch_size, n_agents, action_dim]

        Returns:
            Q-values for each agent's actions
        """
        # Concatenate observations and actions for each agent
        inputs = torch.cat([obs, actions], dim=2)

        # Encode inputs for each agent
        encoded = self.encoder(inputs)

        # Generate keys, queries and values for attention
        keys = self.key_net(encoded)
        queries = self.query_net(encoded)
        values = self.value_net(encoded)

        # Compute scaled dot-product attention
        scale = torch.sqrt(torch.FloatTensor([self.hidden_dim])).item()
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / scale
        attention = F.softmax(scores, dim=-1)

        # Apply attention weights to values
        attended = torch.matmul(attention, values)

        # Generate final Q-values
        q_values = self.final_net(attended)

        return q_values


class Actor(nn.Module):
    """
    Actor network that generates actions for each agent based on local observations.

    Args:
        obs_dim (int): Dimension of observation space
        action_dim (int): Dimension of action space
        hidden_dim (int): Size of hidden layers
    """

    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)  # Output probabilities for discrete actions
        )

    def forward(self, obs):
        return self.net(obs)


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.

    Args:
        capacity (int): Maximum size of buffer
        obs_dim (int): Dimension of observation space
        n_agents (int): Number of agents
        action_dim (int): Dimension of action space
    """

    def __init__(self, capacity, obs_dim, n_agents, action_dim):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.obs_dim = obs_dim
        self.n_agents = n_agents
        self.action_dim = action_dim

    def push(self, state, global_state, action, reward, next_state, next_global_state, done):
        """Add transition to buffer"""
        self.buffer.append((state, global_state, action,
                           reward, next_state, next_global_state, done))

    def sample(self, batch_size):
        """Sample a batch of transitions"""
        batch = random.sample(self.buffer, batch_size)
        state, global_state, action, reward, next_state, next_global_state, done = zip(
            *batch)

        # Convert to PyTorch tensors
        state = torch.FloatTensor(np.array(state))
        global_state = torch.FloatTensor(np.array(global_state))
        action = torch.FloatTensor(np.array(action))
        reward = torch.FloatTensor(np.array(reward)).unsqueeze(-1)
        next_state = torch.FloatTensor(np.array(next_state))
        next_global_state = torch.FloatTensor(np.array(next_global_state))
        done = torch.FloatTensor(np.array(done)).unsqueeze(-1)

        return state, global_state, action, reward, next_state, next_global_state, done

    def __len__(self):
        return len(self.buffer)


class MAAC:
    """
    Multi-Agent Attention Critic main class.
    Coordinates multiple actors and a centralized attention critic.

    Args:
        n_agents (int): Number of agents
        obs_dim (int): Dimension of observation space
        action_dim (int): Dimension of action space
        global_state_dim (int): Dimension of global state
        lr (float): Learning rate
        gamma (float): Discount factor
    """

    def __init__(self, n_agents, obs_dim, action_dim, global_state_dim, lr=0.01, gamma=0.95):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Create actor networks (one per agent)
        self.actors = [Actor(obs_dim, action_dim) for _ in range(n_agents)]
        self.actors_target = [Actor(obs_dim, action_dim)
                              for _ in range(n_agents)]

        # Create critic networks (shared among agents)
        self.critic = AttentionCritic(obs_dim, action_dim, n_agents)
        self.critic_target = AttentionCritic(obs_dim, action_dim, n_agents)

        # Initialize optimizers
        self.actor_optimizers = [optim.Adam(
            actor.parameters(), lr=lr) for actor in self.actors]
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Initialize target networks
        self.update_targets(tau=1.0)

        self.gamma = gamma

    def select_action(self, obs, agent_id, epsilon=0.1):
        """
        Select action for a given agent using epsilon-greedy policy

        Args:
            obs: Agent's observation
            agent_id: ID of the agent
            epsilon: Exploration rate
        """
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

        if random.random() < epsilon:
            action = np.random.randint(0, self.action_dim)
        else:
            with torch.no_grad():
                action_probs = self.actors[agent_id](obs_tensor)
                action = torch.argmax(action_probs).item()

        return action

    def update(self, batch_size, buffer):
        """
        Update actor and critic networks using sampled batch

        Args:
            batch_size: Size of sampled batch
            buffer: Replay buffer to sample from

        Returns:
            tuple: (critic_loss, actor_losses) - Loss values for logging
        """
        if len(buffer) < batch_size:
            return 0, [0] * self.n_agents

        # Sample batch of transitions
        states, global_states, actions, rewards, next_states, next_global_states, dones = buffer.sample(
            batch_size)

        # Update critic
        # Get next actions from target actors
        next_actions = []
        for i in range(self.n_agents):
            next_action_probs = self.actors_target[i](next_states[:, i])
            next_actions.append(next_action_probs)
        next_actions = torch.stack(next_actions, dim=1)

        # Compute target Q-values
        target_q = rewards + (1 - dones) * self.gamma * \
            self.critic_target(next_states, next_actions)
        current_q = self.critic(states, actions)

        # Update critic using TD error
        critic_loss = F.mse_loss(current_q, target_q.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actors
        actor_losses = []
        for i in range(self.n_agents):
            current_actions = actions.clone()
            current_actions[:, i] = self.actors[i](states[:, i])

            # Actor loss is negative of critic value
            actor_loss = -self.critic(states, current_actions).mean()
            actor_losses.append(actor_loss.item())

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()

        # Soft update target networks
        self.update_targets()

        return critic_loss.item(), actor_losses

    def update_targets(self, tau=0.01):
        """Soft update target networks"""
        # Update actor targets
        for actor, actor_target in zip(self.actors, self.actors_target):
            for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data)

        # Update critic target
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data)
