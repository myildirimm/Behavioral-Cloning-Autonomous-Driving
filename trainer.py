import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np



def vehicles_to_tensor(v, i):
    vv = v[:i] + v[i+1:]
    cv = v[i]

    for j in range(len(vv)):
        vv[j] =vv[j].copy()
        vv[j]["x"] -= cv["x"]
        if vv[j]["x"] < 0:
            vv[j]["x"] += vv[j]["width"]
        else:
            vv[j]["x"] -= cv["width"]

        vv[j]["y"] -= cv["y"]
        if vv[j]["y"] < 0:
            vv[j]["y"] += vv[j]["height"]
        else:
            vv[j]["y"] -= cv["height"]

        vv[j]["xVelocity"] -= cv["xVelocity"]
        vv[j]["yVelocity"] -= cv["yVelocity"]

    vv = sorted(vv, key=lambda x: abs(x["x"]**2 + x["y"]**2))[:10]
    # vv = [[x["x"], x["y"], x["width"], x["height"], x["xVelocity"], x["yVelocity"]] for x in vv]
    # list = [[cv["normX"], cv["normY"], cv["width"], cv["height"], cv["xVelocity"], cv["yVelocity"]]] + vv

    vv = [[x["x"], x["y"],  x["xVelocity"], x["yVelocity"]] for x in vv]
    list = [[cv["normX"], cv["normY"], cv["xVelocity"], cv["yVelocity"]]] + vv


    return torch.from_numpy(np.array([[vv]], dtype=np.float32))

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv = nn.ReLU(nn.Conv2d(1, 40, (1,4)))
        self.conv2 = nn.ReLU(nn.Conv2d(40, 40, (1,1)))
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            # nn.Conv2d(1, 30, (1,6)),
            # nn.LazyBatchNorm1d(),
            # nn.ReLU(),         
            # nn.Linear(40, 200),
            # nn.Dropout(p=0.1),
            # nn.LazyBatchNorm1d(),
            # nn.ReLU(),
            # nn.LazyLinear(200),
            # nn.LazyBatchNorm1d(),
            # nn.LazyLinear(200),
            # nn.LazyBatchNorm1d(),
            # nn.Sigmoid(),
            # nn.LazyLinear(100),
            # nn.LazyBatchNorm1d(),
            # nn.Sigmoid(),
            # nn.ReLU(),
            # nn.LazyLinear(100),
            # nn.LazyBatchNorm1d(),
            # nn.Tanh(),
            nn.ReLU(),
            nn.LazyLinear(200),
            nn.ReLU(),
            nn.LazyLinear(2),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.conv2(x)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits



class Trainer:
    def __init__(self, writer_dir='board/'):
        self.model = NeuralNetwork()
        self.learning_rate = 5e-3
        self.batch_size = 64

        self.loss_fn = nn.MSELoss()
        # self.loss_fn = nn.SmoothL1Loss()
        # self.loss_fn = nn.L1Loss()
        self.optimizer = torch.optim.NAdam(self.model.parameters(), lr=self.learning_rate)

        self.writer = SummaryWriter(writer_dir)

        self.start = 1


    def train_loop(self, data_provider, length=100):
        avg_loss = 0
        self.model.train()


        for j in range(self.start, self.start + length +1):
            self.start = j
            X, y = data_provider.get_batch(self.batch_size)

            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            avg_loss += loss.item()

            if j % 50 == 0 and j > 1:
                val_loss = 0
                for k in range(10):
                    X, y = data_provider.get_batch(self.batch_size, validation=True)

                    pred = self.model(X)
                    loss = self.loss_fn(pred, y)
                    val_loss += loss.item()
                    self.writer.add_histogram('predictions', pred, j)

                print(f"Step {j} with loss {avg_loss / 50 } and validation_loss {val_loss / 10}")
                self.writer.add_scalars('Training Loss',
                                    { 'Training' : avg_loss / 50, 'Validation': val_loss/10 },
                                    j)
                
                avg_loss = 0
        
        # self.start += length


    def predict_acceleration(self, v, i):
        self.model.eval()

        t = vehicles_to_tensor(v, i)

        # print(t.numpy().as_list())
        r = self.model(t)
        n = r[0].detach().numpy()

        return float(n[0]), float(n[1])