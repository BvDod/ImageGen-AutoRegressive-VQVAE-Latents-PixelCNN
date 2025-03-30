import torch
import torchvision
from models.vq_vae import VQVAE
import random
import numpy

from models.image_gpt import ImageGPT
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def train_imageGPT(settings):
    writer = SummaryWriter()

    # Load dataset of encoded discrete embeddings
    input_data_train = torch.load(f"datasets/{settings['dataset']}/discrete_train.pt").int()
    input_data_val = torch.load(f"datasets/{settings['dataset']}/discrete_train.pt").int()
    print(f"Data shape: {input_data_train.shape}")

    train = torch.utils.data.TensorDataset(input_data_train)
    val = torch.utils.data.TensorDataset(input_data_train)

    dataloader_train = DataLoader(train, batch_size=settings["batch_size"], shuffle=True, drop_last=True, pin_memory=True)
    dataloader_test = DataLoader(val, batch_size=settings["batch_size"]*4, pin_memory=True)


    device = "cpu"
    print(f"Device: {device}" + "\n")

    # Load model
    model_settings = settings["model_settings"]
    model_settings["num_channels"] =  input_data_train.shape[1]
    model_settings["input_shape"] = input_data_train.shape[-2:]
    model_settings["device"] = device

    model = ImageGPT(model_settings)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
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

        if settings["save_model"]:
            import os
            path = f"models/saved_models/{settings['dataset']}/"
            os.makedirs(path, exist_ok = True) 
            torch.save(model.state_dict(), path + "model_latest.pt")
            if test_losses[-1] < best_test_loss:
                best_test_loss = test_losses[-1]
                torch.save(model.state_dict(), path + f"model_best({epoch}).pt")
                
        model.eval()
        test_losses_epoch = []
        with torch.no_grad():
            for batch_i, (x_test,) in enumerate(dataloader_test):
                x_test = x_test.to(device)
                prediction = model(x_test)

                loss = loss_function(prediction, x_test.long())
                test_losses_epoch.append(loss.item())
            print(f"Test loss: {sum(test_losses_epoch) / len(test_losses_epoch)}")
            test_losses.append(sum(test_losses_epoch) / len(test_losses_epoch))
            writer.add_scalar("Loss/test", test_losses[-1], epoch)
        
        
        if settings["save_model"]:
            import os
            path = f"models/saved_models/{settings['dataset']}/"
            os.makedirs(path, exist_ok = True) 
            torch.save(model.state_dict(), path + "model_latest.pt")
            if test_losses[-1] < best_test_loss:
                best_test_loss = test_losses[-1]
                torch.save(model.state_dict(), path + f"model_best({epoch}).pt")

if __name__ == "__main__":
    settings = {
        "dataset": "celebA",
        "save_model": True,

        #"print_debug": False,
        #"example_image_amount": 4,
        #"save_reconstructions_first_epoch": True,
        "batch_size": 32,
        "learning_rate": 3e-4, # for x-ray
        #"max_epochs": 100000,
        #"early_stopping_epochs": 3,

        "model_settings" : {
            "num_hidden": 128,
            "embedding_dim": 64,
            "num_embeddings": 512,
            "transformer_layers": 5,
            "attention_heads": 4,
            "vocabulary_size": 512
        }
    }
    train_imageGPT(settings)
