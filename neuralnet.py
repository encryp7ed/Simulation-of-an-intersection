import torch
import torch.nn as nn
import torch.optim as optim
from models import *
import numpy as np
import random
from collections import namedtuple


class TrafficLightAgent:
    def __init__(self, model):
        self.model = model
        self.current_observation = None

    def observe(self, observation):
        self.current_observation = observation

    def decide(self, epsilon=0.1):
        if self.current_observation:
            state = torch.tensor(self.current_observation, dtype=torch.float32).unsqueeze(0)
            if np.random.rand() < epsilon:
                action = np.random.choice([0, 1])
            else:
                with torch.no_grad():
                    q_values = self.model(state)
                    action = q_values.argmax().item()
            return action
        else:
            return 0


class TrafficModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TrafficModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

'''
def train_model(model, input_data, labels, avg_time, epochs=500):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(input_data)
        loss = criterion(outputs, labels)

        # Штраф за среднее время больше 9 секунд
        if avg_time > 9:
            loss += 0.1  # Увеличиваем потери на 0.1

        loss.backward()
        optimizer.step()
        # print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}, Avg Time: {avg_time}')
'''


def train_agent(model, optimizer, criterion, states, actions, rewards, next_states, gamma=0.99):
    model.train()

    # Преобразование данных в тензоры
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)

    # Предсказание текущих значений Q
    q_values = model(states)

    # Предсказание максимальных Q значений для следующих состояний
    next_q_values = model(next_states).detach().max(1)[0]

    # Целевые значения Q
    targets = rewards + gamma * next_q_values

    # Выбор Q значений для совершенных действий
    q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    # Вычисление функции потерь
    loss = criterion(q_values, targets)

    # Оптимизация модели
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()  # Возвращаем значение loss


def select_action(model, state, epsilon):
    if np.random.rand() < epsilon:
        # Выбор случайного действия
        return np.random.choice([0, 1])
    else:
        # Выбор действия на основе модели
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = model(state)
            return q_values.argmax().item()


def get_state(roads, traffic_lights):
    state = []
    for road in roads:
        state.append(len(road.cars))
    for light in traffic_lights:
        state.append(light.green_duration)
        state.append(light.red_duration)
    return state


def calculate_reward(previous_state, current_state):
    prev_total_time = sum(Car.times_before_intersection)
    curr_total_time = sum(Car.times_after_intersection)

    # Награда положительная, если времени затрачено меньше
    delta_time = prev_total_time - curr_total_time

    reward = np.clip(delta_time / 100, -1, 1)

    # Очищаем списки после вычисления награды
    Car.times_before_intersection.clear()
    Car.times_after_intersection.clear()
    return reward


def optimize_model(model, optimizer, criterion, replay_memory):
    if len(replay_memory) < BATCH_SIZE:
        return 0
    transitions = replay_memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    state_batch = torch.tensor(batch.state, dtype=torch.float32)
    action_batch = torch.tensor(batch.action, dtype=torch.int64)
    reward_batch = torch.tensor(batch.reward, dtype=torch.float32)
    next_state_batch = torch.tensor(batch.next_state, dtype=torch.float32)

    state_action_values = model(state_batch).gather(1, action_batch.unsqueeze(1))

    next_state_values = model(next_state_batch).max(1)[0].detach()

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


# Определяем функцию потерь
criterion = nn.MSELoss()

# Параметры модели
input_size = 8
hidden_size = 50
output_size = 2


BATCH_SIZE = 64
GAMMA = 0.99
CAPACITY = 10000  # Емкость памяти повторов
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

# Создаем модель нейросети
traffic_model = TrafficModel(input_size, hidden_size, output_size)

# Определяем оптимизатор
optimizer = optim.SGD(traffic_model.parameters(), lr=0.01)