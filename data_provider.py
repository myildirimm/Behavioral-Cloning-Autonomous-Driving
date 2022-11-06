#from socket import CAN_BCM_RX_ANNOUNCE_RESUME
import pandas as pd
import numpy as np
import random
import torch

class DataProvider:
    def __init__(self, max_frame_id = None):
        self.tracks = pd.read_csv('11_tracks.csv')

        if max_frame_id:
            self.tracks = self.tracks[self.tracks.frame < max_frame_id]

        self.tracks = self.tracks[self.tracks.xVelocity > 0]
        self.tracks.xAcceleration -= self.tracks.xAcceleration.mean()
        # self.tracks.xAcceleration -= self.tracks.xAcceleration.mean()
        self.tracks.yAcceleration = self.tracks.yAcceleration * 10

        self.tracks['origWidth'] = self.tracks.width
        self.tracks['origHeight'] = self.tracks.height

        self.tracks['normX'] =  (self.tracks.x  - self.tracks.x.mean()) / self.tracks.x.std()
        self.tracks['normY'] =  (self.tracks.y  - self.tracks.y.mean()) / self.tracks.y.std()

        self.tracks.width = (self.tracks.width  - self.tracks.width.mean()) / self.tracks.width.std()
        self.tracks.height = (self.tracks.height  - self.tracks.height.mean()) / self.tracks.height.std()

        self.number_of_frames = self.tracks.frame.max()

        self.split_frame = int(self.number_of_frames * 0.8)
        self.validation = self.tracks[self.tracks.frame > self.split_frame]
        self.tracks = self.tracks[self.tracks.frame < self.split_frame]

        self.number_of_frames = self.tracks.frame.max()
        self.number_of_frames_validation = self.validation.frame.max()



    def get_random_frame(self, validation=False):
        if validation:
            frame_id = random.randint(self.number_of_frames + 1, self.number_of_frames_validation - 1)

            return self.validation[self.validation.frame == frame_id]
        frame_id = random.randint(0, self.number_of_frames - 1)

        return self.tracks[self.tracks.frame == frame_id]

    def get_random_car_data(self, validation=False):
        frame = self.get_random_frame(validation=validation)
        car_list = frame.id.to_list()
        if len(car_list) == 0:
            return self.get_random_car_data(validation=validation)

        car_id = random.choice(car_list)
        car = frame[frame.id == car_id]

        car_x = float(car.x)
        car_y = float(car.y)
        car_vx = float(car.xVelocity)
        car_vy = float(car.yVelocity)
        car_width = float(car['origWidth'])
        car_height = float(car['origHeight'])

        # reduced_frame = frame[abs(frame.x - car_x) < 70].copy()
        reduced_frame = frame.copy()
        reduced_frame = reduced_frame[reduced_frame.id != car_id][['x', 'y', 'width', 'height', 'xVelocity', 'yVelocity', 'origWidth', 'origHeight']]

        def modify_row(row):
            row["x"] -= car_x
            row["y"] -= car_y

            if row["x"] < 0:
                row["x"] += row["origWidth"]
            if row["x"] > 0:
                row["x"] -= car_width

            if row["y"] < 0:
                row["y"] += row["origHeight"]
            if row["y"] > 0:
                row["y"] -= car_height


            row["xVelocity"] -= car_vx
            row["yVelocity"] -= car_vy

            row["car_vx"] = car_vx
            row["car_vy"] = car_vy

            return row
            

        reduced_frame = reduced_frame.apply(modify_row, axis=1)

        reduced_frame.loc[:, 'dist'] = reduced_frame.x **2 + reduced_frame.y **2
        reduced_frame = reduced_frame.sort_values(by='dist', ascending=1)

        input_1 = car[['normX', 'normY', 'xVelocity', 'yVelocity']].to_numpy()
        input_2 = reduced_frame[['x', 'y', 'xVelocity', 'yVelocity', 'width', 'height', "car_vx", "car_vy"]].to_numpy()
        input_3 = np.zeros((10, 8))

        input_1[0, 0] = 0

        # input_data = np.concatenate((input_1, input_2, input_3))[:10,:]
        input_data = np.concatenate((input_2, input_3))[:10,:]

        # input_data[0, 1] += (np.random.random(1) - 0.5) / 100 # y position -> lane
        # input_data[1:, 0] += (np.random.random((9)) - 0.5) / 10
        # input_data[1:, 1] += (np.random.random((9)) - 0.5) / 10

        # input_data[:, 2] += np.random.random((10)) - 0.5
        # input_data[:, 3] += (np.random.random((10)) - 0.5) / 2

        # input_data[:, 4] += (np.random.random((10)) - 0.5) * 1
        # input_data[:, 5] += (np.random.random((10)) - 0.5) / 100



        return {
            "input": input_data,
            "target": car[['xAcceleration', 'yAcceleration']].to_numpy()[0]
        }

    def get_batch(self, batch_size, validation=False):
        batch = [self.get_random_car_data(validation=validation) for _ in range(batch_size)]
        batch_X = np.array([x["input"] for x in batch], dtype=np.float32)
        batch_Y = np.array([x["target"] for x in batch], dtype=np.float32)

        return torch.from_numpy(batch_X).reshape((-1, 1, 10, 8)), torch.from_numpy(batch_Y)


    def get_vehicle_information(self):
        frame_data = self.get_random_frame()
        frame_data = frame_data[['x', 'y', 'width', 'height', 'xVelocity', 'yVelocity',
       'xAcceleration', 'yAcceleration', 'origWidth', 'origHeight', 'normX', 'normY']]

        if len(frame_data) < 11:
            return self.get_vehicle_information()

        return [x.to_dict() for _, x in frame_data.iterrows()]