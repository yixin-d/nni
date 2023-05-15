import torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader


class TestModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(128, 256)
        self.fc2 = torch.nn.Linear(256, 512)
        self.fc3 = torch.nn.Linear(512, 100)

    def forward(self, inputs: torch.Tensor):
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(inputs)))))


class TrainDataset(Dataset):
    def __init__(self) -> None:
        self.data = torch.rand(1000, 128)
        self.target = torch.rand(self.data.shape[0], 100)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index) -> torch.Tensor:
        assert index < self.data.shape[0]
        return self.data[index], self.target[index]


device = 'cuda'


def train(model: torch.nn.Module, optimizer: torch.optim.Optimizer, training_step,
          lr_scheduler, max_steps: int, max_epochs: int):
    assert max_epochs is not None or max_steps is not None
    dataloader = DataLoader(TrainDataset(), batch_size=32)
    max_steps = max_steps if max_steps else max_epochs * len(dataloader)
    max_epochs = max_steps // len(dataloader) + (0 if max_steps % len(dataloader) == 0 else 1)
    count_steps = 0

    model.train()
    for epoch in range(max_epochs):
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = training_step((data, target), model)
            loss.backward()
            optimizer.step()
            count_steps += 1


def training_step(batch, model: torch.nn.Module):
    output = model(batch[0])
    loss = F.cross_entropy(output, batch[1])
    return loss


import nni
from nni.contrib.compression import TorchEvaluator
from nni.contrib.compression.quantization import QATQuantizer


if __name__ == '__main__':
    model = TestModel().to(device)

    config_list = [{
        'op_types': ['Linear'],
        'quant_dtype': 'int8',
        'target_names': ['_input_'],
        # per channel wise quant input
        'granularity': 'per_channel'
    }, {
        'op_types': ['Linear'],
        'quant_dtype': 'int8',
        'target_names': ['weight'],
        # per block wise quant weight, block size 4 x 16
        'granularity': [4, 16]
    }]

    optimizer = nni.trace(SGD)(model.parameters(), lr=0.001)
    evaluator = TorchEvaluator(train, optimizer, training_step)

    quantizer = QATQuantizer(model, config_list, evaluator, quant_start_step=100)
    _, calibration_config = quantizer.compress(1000, None)
    print(calibration_config)
