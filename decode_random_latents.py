import torch
import torchvision
from models.vq_vae import VQVAE
import random
import numpy


# Load dataset of encoded discrete embeddings
dir = "datasets/x-ray/"
input_data = torch.load(dir + "discrete.pt")

# Load model
model_settings = {
            "encoder_architecture": "VIT",
            "decoder_architecture": "VIT",
            "num_hidden": 128,
            "num_residual_hidden": 128,
            "embedding_dim": 64,
            "num_embeddings": 512,
            "commitment_cost": 0.25,
            "transformer_layers": 5,
            "attention_heads": 4,
            "patch_size": 8,
    }
device = "cpu"
print(f"Device: {device}" + "\n")

# Load model
model_settings["num_channels"] =  3
model_settings["input_shape"] = (256, 256)
model = VQVAE(model_settings=model_settings)
model.load_state_dict(torch.load("models/saved_models/celebA/model_best(222).pt", map_location=torch.device('cpu'), weights_only=True))
model.to(device)

model.eval()
with torch.no_grad():
    ### Generate random latents to decode
    embeddings_to_use = list(input_data.unique())                       # Only use discrete latents that actually occur in the actuial latents
    latent_input = numpy.random.choice(embeddings_to_use, (1,32,32))   # Generate new random latents using discrete values it knows
    latent_input = torch.from_numpy(latent_input)
    latent_input = latent_input.to(device)
    # Decode random latents
    reconstruction = model.decode_latents(latent_input)
    # Show generated samples
    grid = torchvision.utils.make_grid(reconstruction, nrow=3)
    img = torchvision.transforms.ToPILImage()(grid) 
    img.save("random.png")  

    ### Compare with decoded latents from validation dataset
    latent_input = input_data[2:3,:32,:32]
    latent_input = latent_input.to(device)
    # Decode
    reconstruction = model.decode_latents(latent_input)
    # Show generated samples
    grid = torchvision.utils.make_grid(reconstruction, nrow=3)
    img = torchvision.transforms.ToPILImage()(grid) 
    img.save("validation.png") 
