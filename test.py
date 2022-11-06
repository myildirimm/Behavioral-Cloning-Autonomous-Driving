import os
import torch

from data_provider import DataProvider
from trainer import Trainer

trainer = Trainer(writer_dir='board/test/')
d = DataProvider()


if os.path.exists('model.pth'):
    trainer.model = torch.load('model.pth')
else:
    trainer.train_loop(d, 100)

    trainer.writer.flush()
    torch.save(trainer.model, 'model.pth')


from plot_game import ScenarioData

vehicles = d.get_vehicle_information()

scenario_data = ScenarioData(420,36.12,1366,118)
scenario_data.init_scenario(vehicles, trainer.predict_acceleration)
scenario_data.plot_scenario()