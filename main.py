from torchvision.transforms import transforms, ToPILImage
from PIL import Image
import torch
import matplotlib.pyplot as plt
import torchvision
import argparse
import utils
from utils import ddim_config, DDIM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    arguments = utils.load_yml_file(args.config)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config = ddim_config(**arguments)
    ddim = DDIM(config, device)

    org_image = Image.open(config.image_path)

    x_enocoed = ddim.encode(
        image=org_image,
        num_inference_steps=config.num_inference_steps,
        prompt=config.prompt,
        guidance_scale=config.guidance_scale,
    )

    inverted_img = ddim.decode(
        x_enocoed,
        num_inference_steps=config.num_inference_steps,
        strength=config.strength,
        prompt=config.prompt,
        guidance_scale=config.guidance_scale,
    )

    genrated_img = ddim.generate_image(config.edit_prompt)

    images = {
        "org image": org_image,
        "inverted image": inverted_img,
        "generated image": genrated_img,
    }
    utils.visualize_results(images=images, font_size=config.font_size)
