from decimal import Decimal
import os
import pygame
import time
import random

# agent details
# need to retrain after changing those
WIDTH = 15
HEIGHT = 8

INIT_TRACK = 5
INIT_SPEED = 70
INIT_DIRECTION = 1


def random_pygame_color():
    return pygame.Color(int(random.random()*256), int(random.random()*256), int(random.random()*256))

class ScenarioData():
    def __init__(self,road_lenth,road_width,X,Y):
        self.running = True
        self.timestep = 1/25
        self.start_time = 0
        self.t_count = 1
        self.white = (255, 255, 255)
        self.image = pygame.image.load(r'11_highway.jpg')
        self.image = pygame.transform.scale(self.image, (X, Y))
        self.road_lenth = road_lenth
        self.road_width = road_width
        self.X = X
        self.Y = Y


    def init_scenario(self, vehicles, update_func):
        self.update_func = update_func
        self.vehicles = vehicles
        self.vehicle_colors = [random_pygame_color() for _ in self.vehicles]


        x = 0
        y = 300
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (x,y)
        pygame.init()

        X = 1366
        Y = 118
        self.display_surface = pygame.display.set_mode((X, Y))
        pygame.display.set_caption('Image')


    def update_vehicles(self):
        new_vehicles = []

        for idx, v in enumerate(self.vehicles):
            xAccel, yAccel = self.update_func(self.vehicles, idx)
            v = v.copy()
            v["xVelocity"] += xAccel / 250
            v["yVelocity"] += yAccel / 2500

            v["x"] += v["xVelocity"] / 250
            v["y"] += v["yVelocity"] / 250

            if v["x"] > self.road_lenth:
                v["x"] = v["x"] - self.road_lenth
            if v["x"] < 0:
                v["x"] = v["x"] + self.road_lenth

            if v["y"] > self.road_width:
                v["yVelocity"] *= -1
            if v["y"] < 0:
                v["yVelocity"] *= -1


            new_vehicles.append(v)

        self.vehicles = new_vehicles

    def detect_collision(self, polygons):
        polygons = sorted(polygons, key=lambda p: p[0][0])

        for j, p1 in enumerate(polygons):
            for k in range(j+1, len(polygons)):
                p2 = polygons[k]
                if p2[0][0] > p1[1][0]:
                    break
                if p2[0][1] > p1[0][1] and p2[0][1] < p1[2][1]:
                    return True
                if p2[2][1] > p1[0][1] and p2[2][1] < p1[2][1]:
                    return True



        return False


    def plot_scenario(self):
        print('scenario start')
        self.start_time = time.time()

        loop_count = 0

        while self.running:
            loop_count += 1
            self.display_surface.fill(self.white)
            self.display_surface.blit(self.image, (0, 0))

            self.update_vehicles()

            polygons = []

            for vehicle, color in zip(self.vehicles, self.vehicle_colors):
                a = float(vehicle['x']) * self.X / self.road_lenth
                b = (float(vehicle['y'])+0.2) * self.Y / self.road_width
                w = float(vehicle['origWidth']) * self.X / self.road_lenth
                h = float(vehicle['origHeight']) * self.Y / self.road_width

                points = [(a, b), (a+w, b), (a+w, b+h), (a, b+h)]
                polygons.append(points)
                pygame.draw.polygon(self.display_surface, color, points)

            pygame.display.update()

            for event in pygame.event.get() :
                if event.type == pygame.QUIT :
                    self.running = False

            if self.detect_collision(polygons):
                self.running=False
                print(f"COLLISION after {loop_count}")

            self.sleep()

        time.sleep(5)

        pygame.quit()

    def sleep(self):
        local_time = time.time() - self.start_time
        sleep_time = self.t_count * self.timestep - local_time
        if sleep_time > 0.1:
            sleep_time = 0.1
        if sleep_time > 0:
            time.sleep(sleep_time)

    def close(self):
        self.running = False

