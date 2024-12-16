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
        state_dim (int): Dimension of local observation space
        action_dim (int): Dimension of action space
        n_agents (int): Number of agents
        hidden_dim (int): Size of hidden layers
    """

    def __init__(self, state_dim, action_dim, n_agents, hidden_dim=64):
        super(AttentionCritic, self).__init__()

        # Encoder network: processes concatenated state and action
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        # Multi-head attention
        self.n_heads = 4
        self.head_dim = hidden_dim // self.n_heads

        # Networks for attention mechanism
        # Generates keys for attention
        self.key_net = nn.Linear(hidden_dim, hidden_dim)
        # Generates queries for attention
        self.query_net = nn.Linear(hidden_dim, hidden_dim)
        # Generates values for attention
        self.value_net = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(0.1)

        # Final network to produce Q-values
        self.final_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

        self.n_agents = n_agents
        self.hidden_dim = hidden_dim

    def forward(self, states, actions):
        """
        Forward pass of the critic network.

        Args:
            states: Agent observations [batch_size, n_agents, state_dim]
            actions: Agent actions [batch_size, n_agents, action_dim]

        Returns:
            Q-values for each agent's actions
        """
        batch_size = states.shape[0]

        # Each agent gets the same global state
        # states shape: [batch_size, n_agents, state_dim]
        inputs = torch.cat([states, actions], dim=2)
        encoded = self.encoder(inputs)

        # Generate keys, queries and values for multi-head attention
        keys = self.key_net(encoded).view(
            batch_size, self.n_agents, self.n_heads, self.head_dim)
        queries = self.query_net(encoded).view(
            batch_size, self.n_agents, self.n_heads, self.head_dim)
        values = self.value_net(encoded).view(
            batch_size, self.n_agents, self.n_heads, self.head_dim)

        # Compute scaled dot-product attention
        scale = torch.sqrt(torch.FloatTensor(
            [self.head_dim])).to(states.device)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / scale

        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # Apply attention weights to values
        attended = torch.matmul(attention, values)
        attended = attended.view(batch_size, self.n_agents, self.hidden_dim)

        # Generate final Q-values
        q_values = self.final_net(attended)

        return q_values


class Actor(nn.Module):
    """
    Actor network that generates actions for each agent based on the full observations.

    Args:
        state_dim (int): Dimension of local observation space
        action_dim (int): Dimension of action space
        hidden_dim (int): Size of hidden layers        
    """

    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

        # Initialize weights
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                nn.init.zeros_(layer.bias)

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

    def __init__(self, capacity: int, state_dim: int, n_agents: int, action_dim: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.action_dim = action_dim

    def push(self, state: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
             next_state: np.ndarray, done: bool):
        """
        Add a new transition to the buffer.

        Args:
            state: [n_agents, state_dim] 
            actions: [n_agents, action_dim] 
            rewards: [n_agents, 1]   
            next_state: [n_agents, state_dim]
            done: bool
        """
        self.buffer.append((state, actions, rewards, next_state, done))

    def sample(self, batch_size: int):
        """Sample a batch of transitions"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)

        states, actions, rewards, next_states, dones = zip(
            *[self.buffer[idx] for idx in indices])

        # Convert to torch tensors with proper shapes
        # [batch_size, n_agents, state_dim]
        states = torch.FloatTensor(np.array(states))
        # [batch_size, n_agents, action_dim]
        actions = torch.FloatTensor(np.array(actions))
        # [batch_size, n_agents, 1]
        rewards = torch.FloatTensor(np.array(rewards))
        # [batch_size, n_agents, state_dim]
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(
            np.array(dones)).unsqueeze(-1)  # [batch_size, 1]

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class MAAC:
    """
    Multi-Agent Attention Critic main class.
    Coordinates multiple actors and a centralized attention critic.

    Args:
        n_agents (int): Number of agents
        state_dim (int): Dimension of local observation space
        action_dim (int): Dimension of action space
        lr (float): Learning rate
        gamma (float): Discount factor
        tau (float): Soft update rate for target networks
    """

    def __init__(self, n_agents, state_dim, action_dim, lr=3e-4, gamma=0.99, tau=0.005):
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Create actor networks (one per agent)
        self.actors = [Actor(state_dim, action_dim).to(self.device)
                       for _ in range(n_agents)]
        self.actors_target = [Actor(state_dim, action_dim).to(
            self.device) for _ in range(n_agents)]

        # Create critic networks (shared among agents)
        self.critic = AttentionCritic(
            state_dim, action_dim, n_agents).to(self.device)
        self.critic_target = AttentionCritic(
            state_dim, action_dim, n_agents).to(self.device)

        # Initialize optimizers
        self.actor_optimizers = [
            optim.Adam(actor.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
            for actor in self.actors
        ]
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8
        )

        # Initialize target networks
        self.update_targets(tau=1.0)

    def select_action(self, state, agent_id, epsilon=0.1):
        """
        Select action for a given agent using epsilon-greedy policy

        Args:
            state: Local observation for the agent
            agent_id: Index of the agent
            epsilon: Exploration rate
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if random.random() < epsilon:
            action = np.random.randint(0, self.action_dim)
        else:
            with torch.no_grad():
                action_probs = self.actors[agent_id](state_tensor)
                noise = torch.randn_like(action_probs) * 0.1
                action_probs = F.softmax(action_probs + noise, dim=-1)
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
        states, actions, rewards, next_states, dones = buffer.sample(
            batch_size)

        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Update critic
        with torch.no_grad():
            # Get next actions from target actors
            next_actions = torch.stack([
                self.actors_target[i](next_states[:, i])
                for i in range(self.n_agents)
            ], dim=1)

            # Compute target Q-values
            target_q = rewards + (1 - dones) * self.gamma * \
                self.critic_target(next_states, next_actions)

        # Update critic using TD error
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        # Update actors
        actor_losses = []
        for i in range(self.n_agents):
            current_actions = actions.clone()
            current_actions[:, i] = self.actors[i](states[:, i])

            actor_loss = -self.critic(states, current_actions).mean()
            actor_losses.append(actor_loss.item())

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 0.5)
            self.actor_optimizers[i].step()

        # Soft update target networks
        self.update_targets()

        return critic_loss.item(), actor_losses

    def update_targets(self, tau=None):
        """Soft update target networks"""

        tau = tau if tau is not None else self.tau

        # Update actor targets
        for actor, actor_target in zip(self.actors, self.actors_target):
            for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data)
        # Update critic target
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data)
