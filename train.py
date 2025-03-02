import torch
import torchvision
from models.vq_vae import VQVAE
import random
import numpy

from torch.utils.tensorboard import SummaryWriter

def train_autoregressive(settings):
    # Tensorboard for logging
    writer = SummaryWriter()
    # Tensorboard doesnt support dicts in hparam dict so lets unpack
    hpam_dict = {key:value for key,value in settings.items() if not isinstance(value, dict)} | settings["model_settings"]
    writer.add_hparams(hpam_dict, {})

    # Load dataset of encoded discrete embeddings
    input_data = torch.load(settings["input_data"])
    training_split = 0.8
    val_split = 1 - training_split

    # Split data in training and val
    data_train, data_val = input_data[:int(training_split*input_data.shape[0]),:,:,:], input_data[int(val_split*input_data.shape[0]):,:,:,:]

    device = "cuda" if torch.cuda.is_available else "cpu"
    print(f"Device: {device}" + "\n")

    # Load model
    model_settings = setings["model_settings"]
    model = VQVAE(model_settings=model_settings)
    model.load_state_dict(torch.load(model_settings["dir"], weights_only=True))
    model.to(device)

    with torch.no_grad():
        ### Generate random latents to decode
        embeddings_to_use = list(input_data.unique())                       # Only use discrete latents that actually occur in the actuial latents
        latent_input = numpy.random.choice(embeddings_to_use, (9,64,64))   # Generate new random latents using discrete values it knows
        latent_input = torch.from_numpy(latent_input)
        latent_input = latent_input.to(device)
        latent_input = torch.nn.functional.one_hot(latent_input.long(), num_classes=512) # Convert to onehot
        # Decode random latents
        reconstruction, _ = model(latent_input, decode_discrete_mode=True)
        # Show generated samples
        grid = torchvision.utils.make_grid(reconstruction, nrow=3)
        img = torchvision.transforms.ToPILImage()(grid) 
        img.save("random.png")  

        ### Compare with decoded latents from validation dataset
        latent_input = torch.nn.functional.one_hot(input_data[:9,:,:].long(), num_classes=512)
        latent_input = latent_input.to(device)
        # Decode
        reconstruction, _ = model(latent_input, decode_discrete_mode=True)
        # Show generated samples
        grid = torchvision.utils.make_grid(reconstruction, nrow=3)
        img = torchvision.transforms.ToPILImage()(grid) 
        img.save("validation.png") 

if __name__ == "__main__":
    setings = {
        "input_data" : "datasets/x-ray/discrete.pt",
        "model_settings": {
            "num_channels": 1,
            "input_shape": (256,256),
            "num_hidden": 64,
            "num_residual_hidden": 32,
            "embedding_dim": 64,
            "num_embeddings": 512,
            "commitment_cost": 0.5,
            "dir": "models/saved_models/x-ray/model.pt",
        }
        }
    }
