from .simple_placement_env import SimplePlacementEnv
from .ppo_agent import PPOAgent


def main():
    components = [(2, 2), (3, 1), (1, 3)]
    connections = [(0, 1), (1, 2), (0, 2)]
    env = SimplePlacementEnv(area_size=(8, 8), components=components, connections=connections)
    agent = PPOAgent(env)
    agent.train(max_episodes=200)


if __name__ == "__main__":
    main()
