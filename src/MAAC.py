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

        # Shape 체크 추가
        assert states.dim(
        ) == 3, f"Expected states to be 3D tensor, got shape {states.shape}"
        assert actions.dim(
        ) == 3, f"Expected actions to be 3D tensor, got shape {actions.shape}"

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
    Experience replay buffer for storing and sampling episodes.

    Args:
        capacity (int): Maximum size of buffer
        buffer_episodes (list): List to store complete episodes
        current_episode (list): Temporary storage for current episode
        n_agents (int): Number of agents
        action_dim (int): Dimension of action space
    """

    def __init__(self, capacity: int,  n_agents: int, action_dim: int):
        self.capacity = capacity
        self.buffer_episodes = []
        self.current_episode = []
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
        # Store transition in current episode
        self.current_episode.append(
            (state, actions, rewards, next_state, done))

        # If episode is done, store it and start new episode
        if done:
            self.buffer_episodes.append(self.current_episode)
            self.current_episode = []

            # Remove oldest episode if capacity is exceeded
            if len(self.buffer_episodes) > self.capacity:
                self.buffer_episodes.pop(0)

    def sample(self, batch_size: int):
        """Sample batch_size number of complete episodes"""
        # Ensure we have enough episodes
        if len(self.buffer_episodes) < batch_size:
            raise ValueError(
                f"Not enough episodes in buffer. Have {len(self.buffer_episodes)}, need {batch_size}")

        # Randomly select batch_size episodes
        selected_episodes = random.sample(self.buffer_episodes, batch_size)

        # Process all transitions from selected episodes
        all_states = []
        all_actions = []
        all_rewards = []
        all_next_states = []
        all_dones = []

        for episode in selected_episodes:
            # Combine all transitions from the episode
            states, actions, rewards, next_states, dones = zip(*episode)

            # Extend our lists with the episode data
            all_states.extend(states)
            all_actions.extend(actions)
            all_rewards.extend(rewards)
            all_next_states.extend(next_states)
            all_dones.extend(dones)

        # Convert to torch tensors
        states = torch.FloatTensor(np.array(all_states))
        actions = torch.FloatTensor(np.array(all_actions))
        rewards = torch.FloatTensor(np.array(all_rewards))
        next_states = torch.FloatTensor(np.array(all_next_states))
        dones = torch.FloatTensor(np.array(all_dones)).unsqueeze(-1)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the number of complete episodes in buffer"""
        return len(self.buffer_episodes)


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

    def __init__(self, n_agents: int, state_dim: int, action_dim: int,
                 lr: float = 3e-4, gamma: float = 0.99, tau: float = 0.005):
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau

        # Default to CPU for inference
        self.device = torch.device("cpu")
        self.training = False
        print(f"Initialized MAAC on {self.device}")

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

    def get_device(self):
        """Return the current device (cpu or gpu)"""
        return self.device

    def to_training_mode(self):
        """Switch to GPU and set training mode"""
        if not self.training:
            if torch.cuda.is_available():
                new_device = torch.device("cuda")
                print(f"Switching to {new_device} for training")

                # Move all networks to GPU
                for actor in self.actors:
                    actor.to(new_device)
                for actor_target in self.actors_target:
                    actor_target.to(new_device)
                self.critic.to(new_device)
                self.critic_target.to(new_device)

                self.device = new_device
            self.training = True
            print(f"Training mode enabled on {self.device}")

    def to_inference_mode(self):
        """Switch to CPU and set inference mode"""
        if self.training:
            new_device = torch.device("cpu")
            print(f"Switching to {new_device} for inference")

            # Move all networks to CPU
            for actor in self.actors:
                actor.to(new_device)
            for actor_target in self.actors_target:
                actor_target.to(new_device)
            self.critic.to(new_device)
            self.critic_target.to(new_device)

            self.device = new_device
            self.training = False
            print(f"Inference mode enabled on {self.device}")

    def select_action(self, state, agent_id, epsilon=0.1):
        """
        Select action for a given agent using epsilon-greedy policy

        Args:
            state: Local observation for the agent
            agent_id: Index of the agent
            epsilon: Exploration rate
        """
        self.to_inference_mode()  # Ensure we're on CPU for inference
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
        self.to_training_mode()  # Switch to GPU for training

        if len(buffer) < batch_size:
            return 0, [0] * self.n_agents

        # Sample batch of transitions
        states, actions, rewards, next_states, dones = buffer.sample(
            batch_size)

        print("Training - States shape:", states.shape)  # 디버깅용
        print("Training - Actions shape:", actions.shape)  # 디버깅용

        # Move to current device (GPU if training)
        states = states.to(self.device)  # [batch_size, n_agents, state_dim]
        actions = actions.to(self.device)  # [batch_size, n_agents, action_dim]
        rewards = rewards.to(self.device)  # [batch_size, n_agents, 1]
        # [batch_size, n_agents, state_dim]
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)            # [batch_size, 1]

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
