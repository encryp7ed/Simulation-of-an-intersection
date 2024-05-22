import simpy
import matplotlib.pyplot as plt
from neuralnet import *


i = 1  # Глобальная переменная для нумерации машин на всех дорогах


def run_simulation(traffic_lights_model, optimizer, criterion, replay_memory, epsilon):
    env = simpy.Environment()

    traffic_light1 = TrafficLight(env, green_duration=10, red_duration=7)
    traffic_light2 = TrafficLight(env, green_duration=7, red_duration=10, opposite_traffic_light=traffic_light1)

    traffic_light1.state = 'green'
    traffic_light2.state = 'red'

    env.process(traffic_light1.run())
    env.process(traffic_light2.run())

    road1 = Road(env, traffic_light1, length=100, road_name='road1')
    road2 = Road(env, traffic_light1, length=100, road_name='road2')
    road3 = Road(env, traffic_light2, length=100, road_name='road3')
    road4 = Road(env, traffic_light2, length=100, road_name='road4')

    road1.next_road = road2
    road2.next_road = road1
    road3.next_road = road4
    road4.next_road = road3

    total_rewards = 0

    def car_generator(env, road):
        global i
        while True:
            current_time = env.now
            intensity = traffic_intensity(current_time, road.road_name)
            yield env.timeout(intensity)
            car_number = i
            i += 1
            car = Car(env, car_number, road, env.now)
            road.add_car(car)

    def traffic_intensity(current_time, road_name):
        if road_name in ['road1', 'road2']:
            if 600 <= current_time < 1200 or 2500 <= current_time < 3000:
                return 2
            else:
                return 10
        else:
            return 8

    def update_state(env, roads, traffic_lights, replay_memory):
        nonlocal total_rewards, epsilon
        previous_state = [0] * len(roads)

        while True:
            state = get_state(roads, traffic_lights)
            action = select_action(traffic_lights_model, state, epsilon)
            # print(f"Выбранное действие: {action}")

            for tl in traffic_lights:
                tl.take_action(action)

            yield env.timeout(1)

            next_state = get_state(roads, traffic_lights)
            reward = calculate_reward(previous_state, next_state)
            # print(f"Награда: {reward}")

            replay_memory.push(Transition(state, action, reward, next_state))

            loss = optimize_model(traffic_lights_model, optimizer, criterion, replay_memory)
            # print(f"Потеря: {loss}")

            total_rewards += reward
            previous_state = next_state

    env.process(update_state(env, [road1, road2, road3, road4], [traffic_light1, traffic_light2], replay_memory))

    env.process(car_generator(env, road1))
    env.process(car_generator(env, road2))
    env.process(car_generator(env, road3))
    env.process(car_generator(env, road4))

    env.run(until=3600)

    avg_time = sum(Car.total_times) / len(Car.total_times)
    Car.total_times.clear()

    return avg_time, total_rewards


simulation_runs = 1000
average_times = []
total_rewards_list = []

input_size = 8
hidden_size = 50
output_size = 2

traffic_model = TrafficModel(input_size, hidden_size, output_size)
optimizer = optim.SGD(traffic_model.parameters(), lr=0.001)
criterion = nn.MSELoss()

epsilon = 0.1
epsilon_decay = 0.99
min_epsilon = 0.01

replay_memory = ReplayMemory(CAPACITY)

for run in range(simulation_runs):
    avg_time, total_rewards = run_simulation(
        traffic_model, optimizer, criterion, replay_memory, epsilon)
    average_times.append(avg_time)
    total_rewards_list.append(total_rewards)

    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    print(f"Симуляция {run + 1} завершена. Среднее время: {avg_time}")

plt.plot(list(range(1, simulation_runs + 1)), average_times)
plt.xlabel('Запуск симуляции')
plt.ylabel('Среднее время проезда (секунды)')
plt.title('Среднее время проезда за запуск симуляции')
plt.grid(True)
plt.show()

print(f"Среднее время за все симуляции: {average_times}")
print(f"Общее вознаграждение за все симуляции: {total_rewards_list}")