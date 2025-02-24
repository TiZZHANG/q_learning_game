import numpy as np
from collections import defaultdict

class QLearningAgent:
    def __init__(
        self, 
        alpha: float = 0.2, 
        gamma: float = 0.95, 
        epsilon: float = 0.5
    ):
        self.alpha = alpha    # 学习率
        self.gamma = gamma    # 折扣因子
        self.epsilon = epsilon  # 探索率
        self.q_table = defaultdict(lambda: np.zeros(4))
    
    def get_state_key(self, state: tuple) -> tuple:
        return tuple(state)
    
    def choose_action(self, state: tuple) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(4)
        state_key = self.get_state_key(state)
        return np.argmax(self.q_table[state_key])
    
    def update(
        self, 
        state: tuple, 
        action: int, 
        reward: float, 
        next_state: tuple
    ) -> None:
        state_key = self.get_state_key(state)
        next_key = self.get_state_key(next_state)
        
        current_q = self.q_table[state_key][action]
        max_next_q = np.max(self.q_table[next_key])
        td_target = reward + self.gamma * max_next_q
        
        self.q_table[state_key][action] += self.alpha * (td_target - current_q)
    
    def decay_epsilon(self, decay_rate: float = 0.997, min_epsilon: float = 0.05) -> None:
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)