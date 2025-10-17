import gym

class WorldModelTrainer():
    def __init__(self):
        self.env = gym.make("CarRacing-v3", domain_randomize=True)

    def train(self):