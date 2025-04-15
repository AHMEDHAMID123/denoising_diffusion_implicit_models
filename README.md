# Denoising Diffusion Implicit Models (DDIM)

This repository contains an implementation of Denoising Diffusion Implicit Models (DDIM) for image generation and inversion tasks, however it's edited so that it is compitable with the rectified flow model. Denoising diffusion implicit models (DDIMs) is a generalization for DDPMs via a class of non-Markovian diffusion process that still leads to the same training objective of the DDPMs. These non-Markovian process can correspond to a deterministic generative process. DDIMs allow us to perform semantically image interpolation directly in the latent space, reconstruct observations with very low error. In addition to that DDIMs provide a much faster sampling time than DDPMs.




## Acknowledgments
[Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
