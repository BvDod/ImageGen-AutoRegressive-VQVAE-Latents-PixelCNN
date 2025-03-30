# Image Generation: ImageGPT on VQ-VAE latents

In this repository, I generate new high-resolution faces by Implementing an ImageGPT-inspired model to autoregressively predict new latents for a ViT-based VQ-VAE I implemented and trained on the CelebA-HQ dataset in an earlier project. (link to project.)

Since the latent-space of the VQ-VAE is so much smaller than the original image (256x256 vs 32x32), and discretized, auto-regressive prediction and sampling of new samples is a lot more manageable. The newly synthesizsed discretized latent spaces can than be upsampled to high-resolution color-images, by decoding them using the VQ-VAE

## Introduction


## Results: Generating new samples

## Results: Inpainting

##