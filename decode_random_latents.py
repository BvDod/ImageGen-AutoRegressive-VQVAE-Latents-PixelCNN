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
    "num_channels": 1,
    "input_shape": (256,256),
    "num_hidden": 64,
    "num_residual_hidden": 32,
    "embedding_dim": 64,
    "num_embeddings": 512,
    "commitment_cost": 0.5,
}
device = "cuda" if torch.cuda.is_available else "cpu"
print(f"Device: {device}" + "\n")

# Load model
model = VQVAE(model_settings=model_settings)
model.load_state_dict(torch.load("models/saved_models/x-ray/model.pt", weights_only=True))
model.to(device)

with torch.no_grad():
    ### Generate random latents to decode
    embeddings_to_use = list(input_data.unique())                       # Only use discrete latents that actually occur in the actuial latents
    latent_input = numpy.random.choice(embeddings_to_use, (64,64,64))   # Generate new random latents using discrete values it knows
    latent_input = torch.from_numpy(latent_input)
    latent_input = latent_input.to(device)
    latent_input = torch.nn.functional.one_hot(latent_input.long(), num_classes=512) # Convert to onehot
    # Decode random latents
    quantized = model.VQ.discrete_to_quantized(latent_input)
    quantized = torch.movedim(quantized, -1, 1)
    reconstruction = model.decoder(quantized)
    # Show generated samples
    grid = torchvision.utils.make_grid(reconstruction, nrow=8)
    img = torchvision.transforms.ToPILImage()(grid) 
    img.show() 

    ### Compare with decoded latents from validation dataset
    latent_input = torch.nn.functional.one_hot(input_data[:64,:,:].long(), num_classes=512)
    latent_input = latent_input.to(device)
    # Decode
    quantized = model.VQ.discrete_to_quantized(latent_input)
    quantized = torch.movedim(quantized, -1, 1)
    reconstruction = model.decoder(quantized)
    # Show generated samples
    grid = torchvision.utils.make_grid(reconstruction, nrow=8)
    img = torchvision.transforms.ToPILImage()(grid) 
    img.show() 
