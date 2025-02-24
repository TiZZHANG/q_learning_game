import random
import numpy as np

# 游戏常量配置
GRID_SIZE = 15
CELL_SIZE = 40
NUM_FOODS = 3
MIN_DISTANCE = 4
WIN_SCORE = 30
MAX_STEPS = 500

class PacmanEnv:
    def __init__(self):
        self.size = GRID_SIZE
        self.reset()
    
    def _manhattan_distance(self, pos1: tuple, pos2: tuple) -> int:
        return abs(pos1[0]-pos2[0]) + abs(pos1[1]-pos2[1])
    
    def _is_valid_position(self, new_pos: tuple, existing_pos: list) -> bool:
        return all(self._manhattan_distance(new_pos, pos) >= MIN_DISTANCE for pos in existing_pos)
    
    def _generate_positions(self, num: int, exclude: list = []) -> list:
        positions = []
        attempts = 0
        while len(positions) < num and attempts < 1000:
            new_pos = (random.randint(0, self.size-1), random.randint(0, self.size-1))
            if new_pos not in exclude and self._is_valid_position(new_pos, positions):
                positions.append(new_pos)
            attempts += 1
        return positions
    
    def reset(self) -> tuple:
        self.player = (random.randint(0, self.size-1), random.randint(0, self.size-1))
        self.ghost = self._generate_positions(1, exclude=[self.player])[0]
        self.foods = self._generate_positions(NUM_FOODS, exclude=[self.player, self.ghost])
        self.score = 0
        self.steps = 0
        return self._get_state()
    
    def _discretize(self, value: int) -> int:
        return np.clip(value, -3, 3)
    
    def _get_state(self) -> tuple:
        if not self.foods:
            return (0, 0, 0, 0)
        
        nearest_food = min(self.foods, key=lambda p: self._manhattan_distance(p, self.player))
        ghost_dx = self.ghost[0] - self.player[0]
        ghost_dy = self.ghost[1] - self.player[1]
        food_dx = nearest_food[0] - self.player[0]
        food_dy = nearest_food[1] - self.player[1]
        
        return (
            self._discretize(ghost_dx),
            self._discretize(ghost_dy),
            self._discretize(food_dx),
            self._discretize(food_dy)
        )
    
    def step(self, action: int) -> tuple:
        # 玩家移动逻辑
        x, y = self.player
        if action == 0: y = max(0, y-1)
        elif action == 1: y = min(self.size-1, y+1)
        elif action == 2: x = max(0, x-1)
        elif action == 3: x = min(self.size-1, x+1)
        self.player = (x, y)
        self.steps += 1
        
        # 幽灵移动逻辑
        gx, gy = self.ghost
        if random.random() < 0.3:  # 30%概率追踪玩家
            dx = 1 if x > gx else -1 if x < gx else 0
            dy = 1 if y > gy else -1 if y < gy else 0
            if dx != 0 and random.random() < 0.5:
                gx = np.clip(gx + dx, 0, self.size-1)
            else:
                gy = np.clip(gy + dy, 0, self.size-1)
        else:  # 随机移动
            gx = np.clip(gx + random.choice([-1,0,1]), 0, self.size-1)
            gy = np.clip(gy + random.choice([-1,0,1]), 0, self.size-1)
        self.ghost = (gx, gy)
        
        # 奖励计算
        reward = -0.05
        done = False
        
        if self.player in self.foods:
            reward += 20
            self.score += 1
            self.foods.remove(self.player)
            new_food = self._generate_positions(1, exclude=[self.player, self.ghost]+self.foods)
            if new_food:
                self.foods.append(new_food[0])
        
        if self.player == self.ghost:
            reward -= 30
            done = True
            
        if self.score >= WIN_SCORE:
            reward += 100
            done = True
            
        if self.steps >= MAX_STEPS:
            done = True
            
        return self._get_state(), reward, done