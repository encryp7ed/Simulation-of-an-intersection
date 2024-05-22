class TrafficLight:
    def __init__(self, env, green_duration, red_duration, opposite_traffic_light=None):
        self.env = env
        self.green_duration = green_duration
        self.red_duration = red_duration
        self.state = 'green'  # Изначально светофор зелёный
        self.opposite_traffic_light = opposite_traffic_light
        self.action = 0  # 0: ничего не делать, 1: сменить фазу

    def change_phase(self):
        if self.state == 'green':
            self.state = 'red'
        else:
            self.state = 'green'

        if self.opposite_traffic_light:
            self.opposite_traffic_light.state = 'red' if self.state == 'green' else 'green'

    def switch_to_green(self):
        if self.opposite_traffic_light is not None and self.opposite_traffic_light.state == 'red':
            self.state = 'green'
            yield self.env.timeout(self.green_duration)
            self.state = 'red'
        else:
            yield self.env.timeout(0)  # Не ждем, если у противоположного светофора не красный

    def switch_to_red(self):
        if self.opposite_traffic_light is not None and self.opposite_traffic_light.state == 'red':
            yield self.env.timeout(0)  # Не ждем, если у противоположного светофора не красный
        else:
            self.state = 'red'
            yield self.env.timeout(self.red_duration)
            self.state = 'green'

    def take_action(self, action):
        if action == 1:
            self.change_phase()
        # Обновим действие для текущего шага
        self.action = action
        # print(f"Traffic light changed to {self.state} at time {self.env.now}")

    def run(self):
        while True:
            # print(f"Traffic light state: {self.state}, time: {self.env.now}")
            if self.state == 'green':
                yield self.env.timeout(self.green_duration)
            else:
                yield self.env.timeout(self.red_duration)
            self.change_phase()
            # print(f"Changed traffic light state to: {self.state}, time: {self.env.now}")


class Road:
    def __init__(self, env, traffic_light, length, road_name):
        self.env = env
        self.traffic_light = traffic_light
        self.length = length
        self.road_name = road_name
        self.cars = []

    def add_car(self, car):
        if car not in self.cars:
            self.cars.append(car)

    def remove_car(self, car):
        if car in self.cars:
            self.cars.remove(car)


class Car(object):
    """Модель машины"""
    MAX_SPEED_KPH = 50  # Максимальная скорость машины в км/ч
    MAX_SPEED_MPS = MAX_SPEED_KPH * (1000 / 3600)  # Переводим скорость в м/с
    ACCELERATION = 2  # Ускорение машины в м/с^2

    total_times = []  # Статическое поле для хранения времени проезда всех машин
    times_before_intersection = []
    times_after_intersection = []

    def __init__(self, env, car_number, road, arrival_time):
        self.env = env
        self.road = road
        self.car_number = car_number  # Номер машины
        self.arrival_time = arrival_time  # Время прибытия машины
        self.total_time = 0  # Время проезда перекрестка
        self.time_before_intersection = 0
        self.time_after_intersection = 0
        self.speed = self.MAX_SPEED_MPS
        self.distance_to_intersection = road.length
        self.distance_to_car_ahead = road.length
        self.process = env.process(self.run())
        self.crossed_intersection = False  # Флаг для отслеживания прохождения перекрестка

    def __str__(self):
        return f"Car on Road {self.road}"

    def run(self):
        while not self.crossed_intersection:
            for car_ahead in self.road.cars:
                if car_ahead != self:
                    distance_to_car_ahead = car_ahead.distance_to_intersection
                    if distance_to_car_ahead < self.speed:
                        self.speed = distance_to_car_ahead
                        yield self.env.timeout(1)

            if self.distance_to_intersection > 0:
                self.distance_to_intersection -= self.speed
                yield self.env.timeout(1)
                if self.road.traffic_light.state == 'red' and self.distance_to_intersection <= 0:
                    self.speed = 0
                    while self.road.traffic_light.state == 'red':
                        yield self.env.timeout(1)

            self.time_before_intersection = self.env.now - self.arrival_time

            if self.road.traffic_light.state == 'green':
                if not self.crossed_intersection:
                    self.road.remove_car(self)
                    self.crossed_intersection = True

                next_road = self.road.next_road
                next_road.add_car(self)
                self.road = next_road
                self.distance_to_intersection = self.road.length

                while self.distance_to_intersection > 0:
                    yield self.env.timeout(1)
                    if self.speed < self.MAX_SPEED_MPS:
                        self.speed += self.ACCELERATION

                    self.distance_to_intersection -= self.speed

        self.total_time = self.env.now - self.arrival_time
        self.time_after_intersection = self.total_time - self.time_before_intersection
        Car.total_times.append(self.total_time)
        Car.times_before_intersection.append(self.time_before_intersection)
        Car.times_after_intersection.append(self.time_after_intersection)
        self.road.remove_car(self)
