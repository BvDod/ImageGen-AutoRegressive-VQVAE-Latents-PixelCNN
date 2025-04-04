# Image Generation: ImageGPT on VQ-VAE latents

<img src="https://github.com/user-attachments/assets/1b86ac91-1306-4e87-843a-a58213b64866" alt="description" width="50%" />

In this repository, I generate new high-resolution faces by implementing an ImageGPT-inspired model to autoregressively predict new latents for a ViT-based VQ-VAE I implemented and trained on the CelebA-HQ dataset in an earlier project ([Link](https://github.com/BvDod/Vector-Quantized-ViT-VAE-Image-Reconstruction))

Since the latent space of the VQ-VAE is so much smaller than the original image (256x256 vs 32x32), and discretized, auto-regressive prediction and sampling of new samples is much more manageable. The newly synthesized discretized latent spaces can then be upsampled to high-resolution color images, by decoding them using the VQ-VAE

## Introduction
TODO

See *models/imagegpt.py* for the autoregressive model. *sample_imageGPT.py* is used to generate completely new samples, while *sample_imageGPT_finish.py* is used only to keep 50% of the original validation image, and autoregressively finish the image.

## Results
Note: these results are NOT from the final model, and quality will still improve a bit more, especially when it comes to detail/ sharpness of the generated images.
### Generating new samples
These samples are fully auto-regressively generated with a temperature of 1.0.

![generated](https://github.com/user-attachments/assets/1b86ac91-1306-4e87-843a-a58213b64866)

### Inpainting
In the following samples, only the top half of the images from the validation set were used to initialize the auto-regressive process, which was thus auto-regressively completed into a full image.

![finished](https://github.com/user-attachments/assets/20db894d-95c8-41ab-a1c4-2a8082744711)



## Next steps:
The autoregressive training (and sampling) currently works in a causal scan-line process. Next, I would like to implement other training and sampling methods, such as Bert-like masking. MaskGIT is also very interesting, which seems to iteravely improve image quality by refineing masked regions of the image.
