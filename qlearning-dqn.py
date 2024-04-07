'''
Author: Tangxuan
Date: 2024-03-26 19:31:33
LastEditTime: 2024-03-28 23:22:43
LastEditors: Tangxuan
Description: Created By Tangxuan And Be Protected
'''
import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

class MyWarehouseEnv(gym.Env):
    def __init__(self):
        # 定义状态空间和动作空间
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, -5, -5, -5]),
                                            high=np.array([29, 29, 29, 5, 5, 5]),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(7)
        
        # 环境参数
        self.grid_size = (30, 30, 30)  # 三维网格世界大小
        self.drone_max_speed = 3  # 无人机最大速度
        self.drone_max_accel = 1  # 无人机最大加速度
        self.battery_capacity = 2000  # 电池容量
        self.battery_consumption_rate = 1  # 电量消耗率
        self.collision_penalty = -1000000  # 碰撞惩罚分
        self.out_of_bound_penalty = -1000000  # 超出边界惩罚分
        #始终点
        self.start_point = (0,0,0)
        self.end_point = (29,29,29)
        # 生成障碍物
        # self.obstacles = self._generate_obstacles(obstacle_density=0.1)
        self.obstacles = self._generate_random_cubes([30,30,30], 20, 3, 5)
        np.save("obstacles.npy",self.obstacles)
        self.obstacles = np.load("obstacles.npy")
        # 初始化无人机状态和目标状态
        self.drone_state = self._reset_drone()
        self.target_state = self._reset_target()
    def drone_hit_obstacle(self):
        if self.obstacles[self.drone_state[0]][self.drone_state[1]][self.drone_state[2]] == True:
            return True
        return False
    def step(self, action):
        # 根据动作更新无人机状态
        x, y, z, vx, vy, vz = self.drone_state
        ax, ay, az = self._decode_action(action)
        
        vx = np.clip(vx + ax, -self.drone_max_speed, self.drone_max_speed)
        vy = np.clip(vy + ay, -self.drone_max_speed, self.drone_max_speed)
        vz = np.clip(vz + az, -self.drone_max_speed, self.drone_max_speed)
        
        x = np.clip(x + vx, 0, self.grid_size[0] - 1)
        y = np.clip(y + vy, 0, self.grid_size[1] - 1)
        z = np.clip(z + vz, 0, self.grid_size[2] - 1)
        
        self.drone_state = (x, y, z, vx, vy, vz)
        
        # 计算即时奖赏
        reward = self._get_reward()
        
        # 检查是否终止
        done = self._is_done()
        
        # 更新电池电量
        self.battery_capacity -= self.battery_consumption_rate * (abs(ax) + abs(ay) + abs(az))
        
        return self.drone_state, reward, done, {}

    def _get_reward(self):
        x, y, z, _, _, _ = self.drone_state
        tx, ty, tz = self.target_state
        
        # 计算距离惩罚
        distance_penalty = np.sqrt((x - tx) ** 2 + (y - ty) ** 2 + (z - tz) ** 2)
        
        # 检查是否碰撞
        if self.obstacles[x, y, z]:
            return self.collision_penalty
        
        # 检查是否超出边界
        if x < 0 or y < 0 or z < 0 or x > 29 or y > 29 or z > 29:
            return self.out_of_bound_penalty
        
        # 到达目标位置获得大奖励
        if distance_penalty < 2:
            return 100
        
        # 其他情况根据距离给予惩罚
        return -distance_penalty

    def _is_done(self):
        x, y, z, _, _, _ = self.drone_state
        tx, ty, tz = self.target_state
        
        # 到达目标位置或电池耗尽时终止
        if np.sqrt((x - tx) ** 2 + (y - ty) ** 2 + (z - tz) ** 2) < 2 or self.battery_capacity <= 0:
            return True
        
        return False

    def _decode_action(self, action):
        # 将离散动作解码为三维加速度
        if action == 0:
            return 0, 0, 0
        elif action == 1:
            return self.drone_max_accel, 0, 0
        elif action == 2:
            return -self.drone_max_accel, 0, 0
        elif action == 3:
            return 0, self.drone_max_accel, 0
        elif action == 4:
            return 0, -self.drone_max_accel, 0
        elif action == 5:
            return 0, 0, self.drone_max_accel
        elif action == 6:
            return 0, 0, -self.drone_max_accel

    def _generate_random_cubes(self, grid_size, num_cubes, min_size, max_size):
        obstacles = np.zeros(grid_size, dtype=bool)
        for _ in range(num_cubes):
            size = np.random.randint(min_size, max_size, size=3)
            position = np.random.randint(0, np.array(self.grid_size) - size, size=3)
            obstacles[position[0]:position[0]+size[0], position[1]:position[1]+size[1], position[2]:position[2]+size[2]] = True
        # 避免在三维空间的最外层生成立方体
        obstacles[0, :, :] = obstacles[-1, :, :] = obstacles[:, 0, :] = obstacles[:, -1, :] = obstacles[:, :, 0] = obstacles[:, :, -1] = False
        obstacles[self.start_point[0],self.start_point[1],self.start_point[2]] = False
        obstacles[self.end_point[0],self.end_point[1],self.end_point[2]] = False
        return obstacles

    def _reset_drone(self):
        x = self.start_point[0]
        y = self.start_point[1]
        z = self.start_point[2]
        vx = 0
        vy = 0
        vz = 0
        return (x, y, z, vx, vy, vz)

    def _reset_target(self):
        tx=self.end_point[0]
        ty=self.end_point[1]
        tz=self.end_point[2]
        return (tx, ty, tz)

    def reset(self):
        # 重置环境状态
        self.drone_state = self._reset_drone()
        self.target_state = self._reset_target()
        self.battery_capacity = 2000
        # self.obstacles = self._generate_obstacles(obstacle_density=0.1)
        # self.obstacles = self._generate_random_cubes([30,30,30], 20, 3, 5)
        self.obstacles = np.load("obstacles.npy")
        return self.drone_state

    def render(self, mode='human'):
        # 可视化环境(简单版本)
        print(f"Drone State: {self.drone_state}")
        print(f"Target State: {self.target_state}")
        print(f"Battery: {self.battery_capacity}")

# 定义状态空间大小和动作空间大小
STATE_SPACE_SIZE = 30 * 30 * 30 * 6  # 6维状态:x,y,z,vx,vy,vz
ACTION_SPACE_SIZE = 7  # 7个离散动作

# 初始化Q表格
Q_TABLE = np.zeros((22*STATE_SPACE_SIZE, ACTION_SPACE_SIZE))

# 给Q-Learning算法设置一些超参数
LEARNING_RATE = 0.1  
DISCOUNT_FACTOR = 0.90
EPSILON = 1.0  # 初始探索率
EPSILON_DECAY = 0.995  # 探索率衰减系数
EPSILON_MIN = 0.10  # 最小探索率

# Q-Learning训练函数
def q_learning(env, num_episodes):
    global EPSILON
    for episode in range(num_episodes):
        # 初始化回合
        state = env.reset()  
        state_encoded = encode_state(state)  # 将状态编码为一维索引
        done = False
        episode_reward = 0

        while not done:
            # 根据当前状态选取动作(ε-greedy)
            if np.random.uniform() < EPSILON:
                action = env.action_space.sample()  # 随机探索动作
            else:
                action = np.argmax(Q_TABLE[state_encoded])  # 选取Q值最大的动作
            
            # 执行动作,获取反馈
            next_state, reward, done, _ = env.step(action)
            next_state_encoded = encode_state(next_state)
            
            # 更新Q表格
            Q_TABLE[state_encoded, action] += LEARNING_RATE * (
                reward + DISCOUNT_FACTOR * np.max(Q_TABLE[next_state_encoded]) - Q_TABLE[state_encoded, action]
            )
            
            state_encoded = next_state_encoded
            episode_reward += reward

        # 更新探索率
        EPSILON = max(EPSILON * EPSILON_DECAY, EPSILON_MIN)
        
        # 输出当前回合的累计奖赏
        print(f"Episode {episode+1}: Reward = {episode_reward}")

# 状态编码函数(将6维状态编码为一维索引)
def encode_state(state):
    x, y, z, vx, vy, vz = state
    return ((((x * 30 + y) * 30 + z) * 5 + vx) * 5 + vy) * 5 + vz

def test_flight_path(env, Q_table):
    state = env.reset()
    done = False
    path = [state]

    while not done:
        state_encoded = encode_state(state)
        action = np.argmax(Q_table[state_encoded])
        state, reward, done, _ = env.step(action)
        path.append(state)

    return path
# env = gym.make("MyWarehouseEnv")  # 自定义Gym环境
env = MyWarehouseEnv()  # 自定义Gym环境

def visualize_path(path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs = [state[0] for state in path]
    ys = [state[1] for state in path]
    zs = [state[2] for state in path]

    ax.plot(xs, ys, zs, label='Drone flight path')
    ax.scatter(xs, ys, zs, c='r', marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
def visualize_path_and_obstacles(path, obstacles):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制飞行路径
    xs = [state[0] for state in path]
    ys = [state[1] for state in path]
    zs = [state[2] for state in path]

    ax.plot(xs, ys, zs, label='Drone flight path')
    ax.scatter(xs, ys, zs, c='r', marker='o')

    # 绘制障碍物
    obs_xs, obs_ys, obs_zs = np.where(obstacles)
    ax.scatter(obs_xs, obs_ys, obs_zs, c='b', marker='s')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


def update(num, data, line):
    line.set_data(data[:2, :num])
    line.set_3d_properties(data[2, :num])

def visualize_path_and_obstacles_animation(path, obstacles):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制障碍物
    obs_xs, obs_ys, obs_zs = np.where(obstacles)
    ax.scatter(obs_xs, obs_ys, obs_zs, c='b', marker='s')

    # 准备飞行路径数据
    data = np.array(path).T
    line, = ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1])

    # 创建动画
    ani = animation.FuncAnimation(fig, update, frames=len(path), fargs=(data, line), interval=100)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()



#########################################################
###############DQN#######################################
#########################################################
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import tqdm
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, gamma=0.90, epsilon=1.0, lr=0.001, batch_size=64, memory_size=10000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.memory = deque(maxlen=memory_size)

        self.q_network = DQN(state_dim, action_dim).to(self.device)
        self.target_network = DQN(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

    def select_action(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                q_values = self.q_network(state)
                return np.argmax(q_values.cpu().numpy())
        else:
            return random.choice(range(self.action_dim))

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        state, action, reward, next_state, done = zip(*batch)

        state = torch.FloatTensor(state).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        q_values = self.q_network(state)
        next_q_values = self.target_network(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        loss = (q_value - expected_q_value.detach()).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

def train_dqn(episodes):
    env = MyWarehouseEnv()
    agent = DQNAgent(state_dim=6, action_dim=7)
    for episode in tqdm.tqdm(range(episodes)):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.train()
            state = next_state
        if episode % 10 == 0:
            agent.update_target_network()
    # 保存模型
    torch.save(agent.q_network.state_dict(), 'dqn_model.pth')

def test_dqn(episodes):
    env = MyWarehouseEnv()
    # 加载模型
    agent = DQNAgent(state_dim=6, action_dim=7)
    agent.q_network.load_state_dict(torch.load('dqn_model.pth'))
    agent.q_network.eval()
    for episode in range(episodes):
        state = env.reset()
        done = False
        drone_states = [state]  # 记录无人机的状态
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            drone_states.append(next_state)  # 记录无人机的新状态
            state = next_state
        # 可视化飞行路径
        # visualize_path_and_obstacles_animation(drone_states, env.obstacles)
    # 可视化飞行路径
    visualize_path_and_obstacles_animation(drone_states, env.obstacles)
if __name__ == "__main__":
    modelist = ["DQN-Test","DQN-Training","Q-Learning-Test","Q-Learning-Training"]
    mode = modelist[3]
    if mode == "Q-Learning-Training":
        print("Using Q-Learning algorithm Training...")
        q_learning(env, num_episodes=10000)
        np.save("Q_table.npy", Q_TABLE)
        Q_table = np.load("Q_table.npy")
        path = test_flight_path(env, Q_table)
        visualize_path_and_obstacles_animation(path, env.obstacles)
    elif mode == "Q-Learning-Test":
        print("Using Q-Learning algorithm Testing...")
        Q_table = np.load("Q_table.npy")
        path = test_flight_path(env, Q_table)
        visualize_path_and_obstacles_animation(path, env.obstacles)
    elif mode == "DQN-Training":
        print("Using DQN Training algorithm...")
        train_dqn(episodes=10000)
        test_dqn(episodes=1)
    elif mode == "DQN-Test":
        print("Using DQN Testing algorithm...")
        test_dqn(episodes=1)
    

    