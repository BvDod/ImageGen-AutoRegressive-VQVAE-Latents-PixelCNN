import torch
import torchvision
from models.vq_vae import VQVAE
import random
import numpy

from models.image_gpt import ImageGPT
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter



writer = SummaryWriter()

# Load dataset of encoded discrete embeddings
input_data_train = torch.load("discrete_train.pt").int()
input_data_val = torch.load("discrete_train.pt").int()
print(f"Data shape: {input_data_train.shape}")

train = torch.utils.data.TensorDataset(input_data_train)
val = torch.utils.data.TensorDataset(input_data_train)

dataloader_train = DataLoader(train, batch_size=32, shuffle=True, drop_last=True, pin_memory=True)
dataloader_test = DataLoader(val, batch_size=32, pin_memory=True)

# Load model
model_settings = {
            "num_hidden": 128,
            "num_residual_hidden": 128,
            "embedding_dim": 64,
            "num_embeddings": 512,
            "commitment_cost": 0.25,
            "transformer_layers": 5,
            "attention_heads": 4,
            "patch_size": 8,
            "vocabulary_size": 512
    }

device = "cpu"
print(f"Device: {device}" + "\n")

# Load model
model_settings["num_channels"] =  input_data_train.shape[1]
model_settings["input_shape"] = input_data_train.shape[-2:]
model_settings["device"] = device

model = ImageGPT(model_settings)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_function = torch.nn.CrossEntropyLoss()


train_losses, test_losses = [], []
best_test_loss = float("inf")

for epoch in range(10000):
    train_losses_epoch = []
    print(f"Epoch: {epoch}")
    
    model.train()
    for batch_i, (x_train,) in enumerate(dataloader_train):
        x_train = x_train.to(device)
        prediction = model(x_train)
        loss = loss_function(prediction, x_train.long())
        loss.backward()

        optimizer.step()
        print(loss)
        optimizer.zero_grad()
        train_losses_epoch.append(loss.item())
    print(f"Train loss: {sum(train_losses_epoch) / len(train_losses_epoch)}")
    train_losses.append(sum(train_losses_epoch) / len(train_losses_epoch))
    writer.add_scalar("Loss/train", train_losses[-1], epoch)

    model.eval()
    test_losses_epoch = []
    with torch.no_grad():
        for x_test, y_test in dataloader_test:
            x_train = x_train.to(device)
            prediction = model(x_train)

            loss = loss_function(prediction, x_train.long())
            test_losses_epoch.append(loss.item())
        test_losses.append(sum(test_losses_epoch) / len(test_losses_epoch))
        writer.add_scalar("Loss/test", test_losses[-1], epoch)