# Image Generation: ImageGPT on VQ-VAE latents

In this repository, I generate new high-resolution faces by implementing an ImageGPT-inspired model to autoregressively predict new latents for a ViT-based VQ-VAE I implemented and trained on the CelebA-HQ dataset in an earlier project ([Link](https://github.com/BvDod/Vector-Quantized-ViT-VAE-Image-Reconstruction))

Since the latent space of the VQ-VAE is so much smaller than the original image (256x256 vs 32x32), and discretized, auto-regressive prediction and sampling of new samples is much more manageable. The newly synthesized discretized latent spaces can then be upsampled to high-resolution color images, by decoding them using the VQ-VAE

## Introduction
TODO

See *models/imagegpt.py* for the autoregressive model. *sample_imageGPT.py* is used to generate completely new samples, while *sample_imageGPT_finish.py* is used only to keep 50% of the original validation image, and autoregressively finish the image.

## Results: Generating new samples

## Results: Inpainting

## Next steps:
The autoregressive training (and sampling) currently works in a causal scan-line process. Next, I would like to implement other training and sampling methods, such as Bert-like masking. MaskGIT is also very interesting, which seems to iteravely improve image quality by refineing masked regions of the image.
