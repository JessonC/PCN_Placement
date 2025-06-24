import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        x = self.base(x)
        probs = self.actor(x)
        value = self.critic(x)
        return probs, value

class PPOAgent:
    def __init__(self, env, lr=3e-4, gamma=0.99, clip_eps=0.2, k_epochs=4, gae_lambda=0.95, ent_coef=0.01):
        self.env = env
        state_dim = np.prod(env.observation_space.shape)
        action_dim = env.action_space.n
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.k_epochs = k_epochs
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef

        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0)
        probs, value = self.policy(state_tensor)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value.squeeze(0), dist.entropy()

    def compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0
        values = values + [0]
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        return advantages

    def update(self, memory):
        states = torch.FloatTensor(np.array(memory['states'])).reshape(len(memory['states']), -1)
        actions = torch.LongTensor(memory['actions']).unsqueeze(-1)
        old_logps = torch.stack(memory['log_probs']).unsqueeze(-1)
        returns = torch.FloatTensor(memory['returns']).unsqueeze(-1)
        advantages = torch.FloatTensor(memory['advantages']).unsqueeze(-1)

        for _ in range(self.k_epochs):
            probs, values = self.policy(states)
            dist = Categorical(probs)
            new_logps = dist.log_prob(actions.squeeze(-1)).unsqueeze(-1)
            entropy = dist.entropy().mean()

            ratios = torch.exp(new_logps - old_logps)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values, returns)
            loss = actor_loss + 0.5 * critic_loss - self.ent_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self, max_episodes=1000, update_timestep=200):
        timestep = 0
        memory = {k: [] for k in ['states', 'actions', 'log_probs', 'rewards', 'dones', 'values']}
        for episode in range(max_episodes):
            state = self.env.reset()
            done = False
            ep_reward = 0
            while not done:
                action, logp, value, _ = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                memory['states'].append(state)
                memory['actions'].append(action)
                memory['log_probs'].append(logp)
                memory['values'].append(value.item())
                memory['rewards'].append(reward)
                memory['dones'].append(done)
                state = next_state
                ep_reward += reward
                timestep += 1
                if timestep % update_timestep == 0 or done:
                    returns = []
                    discounted = 0
                    for r, d in zip(reversed(memory['rewards']), reversed(memory['dones'])):
                        if d:
                            discounted = 0
                        discounted = r + self.gamma * discounted
                        returns.insert(0, discounted)
                    advantages = self.compute_gae(memory['rewards'], memory['values'], memory['dones'])
                    memory['returns'] = returns
                    memory['advantages'] = advantages
                    self.update(memory)
                    for k in memory:
                        memory[k] = []
                    timestep = 0
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode+1}, reward: {ep_reward:.2f}")
