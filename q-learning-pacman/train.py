import pygame
import numpy as np
from game_env import PacmanEnv, GRID_SIZE, CELL_SIZE, WIN_SCORE, MAX_STEPS
from q_learning import QLearningAgent

# 颜色
COLORS = {
    'background': (0, 0, 0),
    'grid': (255, 255, 255),
    'player': (255, 255, 0),
    'ghost': (255, 0, 0),
    'food': (0, 255, 0),
    'text': (255, 255, 255),
    'victory': (0, 255, 0), 
    'defeat': (255, 0, 0)
}

def initialize_pygame():
    pygame.init()
    screen = pygame.display.set_mode(
        (GRID_SIZE*CELL_SIZE, GRID_SIZE*CELL_SIZE + 50))
    pygame.display.set_caption("Pacman Q-Learning")
    return screen

def draw_game_state(screen, env, font, done):
    screen.fill(COLORS['background'])
    
    # 绘制网格
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            rect = pygame.Rect(x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, COLORS['grid'], rect, 1)
    
    # 绘制玩家
    pygame.draw.circle(
        screen, COLORS['player'],
        (env.player[0]*CELL_SIZE + CELL_SIZE//2,
         env.player[1]*CELL_SIZE + CELL_SIZE//2),
        CELL_SIZE//3
    )
    
    # 绘制幽灵
    pygame.draw.rect(
        screen, COLORS['ghost'],
        (env.ghost[0]*CELL_SIZE,
         env.ghost[1]*CELL_SIZE,
         CELL_SIZE, CELL_SIZE)
    )
    
    # 绘制豆子
    for food in env.foods:
        pygame.draw.circle(
            screen, COLORS['food'],
            (food[0]*CELL_SIZE + CELL_SIZE//2,
             food[1]*CELL_SIZE + CELL_SIZE//2),
            CELL_SIZE//4
        )
    
    # 绘制信息面板
    info_surface = pygame.Surface((GRID_SIZE*CELL_SIZE, 50))
    info_surface.fill(COLORS['background'])
    texts = [
        f"Score: {env.score}/{WIN_SCORE}",
        f"Steps: {env.steps}/{MAX_STEPS}",
        f"Foods: {len(env.foods)}"
    ]
    for i, text in enumerate(texts):
        text_surf = font.render(text, True, COLORS['text'])
        info_surface.blit(text_surf, (10 + i*200, 15))
    screen.blit(info_surface, (0, GRID_SIZE*CELL_SIZE))
    
    # 游戏结束时的结算信息
    if done:
        if env.score >= WIN_SCORE:
            result_text = "Victory! Final Score: {}".format(env.score)
            color = COLORS['victory']
        else:
            result_text = "Game Over! Final Score: {}".format(env.score)
            color = COLORS['defeat']
        
        text_surf = font.render(result_text, True, color)
        text_rect = text_surf.get_rect(center=(GRID_SIZE*CELL_SIZE//2, GRID_SIZE*CELL_SIZE//2))
        screen.blit(text_surf, text_rect)

def train_agent(env, agent, episodes=3000):
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward
        
        agent.decay_epsilon()
        
        if ep % 200 == 0:
            print(f"Episode {ep:04d} | Score: {env.score:03d} | "
                  f"Total Reward: {total_reward:6.1f} | "
                  f"Epsilon: {agent.epsilon:.2f}")

def main():
    # 初始化环境
    env = PacmanEnv()
    agent = QLearningAgent()
    
    # 训练阶段
    print("=== Training Start ===")
    train_agent(env, agent)
    print("=== Training Complete ===")
    
    # 可视化演示
    screen = initialize_pygame()
    font = pygame.font.Font(None, 28)
    clock = pygame.time.Clock()
    
    state = env.reset()
    done = False
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        
        # 绘制游戏状态
        draw_game_state(screen, env, font, done)
        
        # 游戏逻辑
        if not done:
            action = np.argmax(agent.q_table[agent.get_state_key(state)])
            state, _, done = env.step(action)
            clock.tick(10)
        else:
            # 游戏结束后等待2秒并重置
            pygame.display.flip()
            pygame.time.wait(2000)
            state = env.reset()
            done = False
        
        pygame.display.flip()

if __name__ == "__main__":
    main()