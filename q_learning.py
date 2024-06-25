import numpy as np
import json

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.6, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state, possible_actions):
        if np.random.rand() < self.epsilon:
            return np.random.choice(possible_actions)
        else:
            q_values = [self.get_q_value(state, action) for action in possible_actions]
            max_q_value = max(q_values)
            return possible_actions[q_values.index(max_q_value)]

    def update_q_value(self, state, action, reward, next_state, possible_actions):
        old_q_value = self.get_q_value(state, action)
        future_q_values = [self.get_q_value(next_state, next_action) for next_action in possible_actions]
        max_future_q_value = max(future_q_values)
        new_q_value = (1 - self.alpha) * old_q_value + self.alpha * (reward + self.gamma * max_future_q_value)
        self.q_table[(state, action)] = new_q_value

    def save_q_table(self, filename='q_table.json'):
        with open(filename, 'w') as f:
            json.dump(self.q_table, f)

    def load_q_table(self, filename='q_table.json'):
        with open(filename, 'r') as f:
            self.q_table = json.load(f)
