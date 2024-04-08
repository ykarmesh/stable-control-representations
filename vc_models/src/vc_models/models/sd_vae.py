import torch

from diffusers import AutoencoderKL


class VaeRepresentation(torch.nn.Module):
    def __init__(
            self,
            model_name_or_path,
            input_image_size,
            flatten=True,
            dtype="float16",
            device="cuda",
            multi_layer=False,
        ):
        super().__init__()
        dtype = torch.float16 if dtype == "float16" else torch.float32
        self.model_name_or_path = model_name_or_path
        self.dtype = dtype
        self.device = device
        self.flatten = flatten
        self.multi_layer = multi_layer

        self.vae = AutoencoderKL.from_pretrained(
            model_name_or_path,
            subfolder="vae"
        ).to(device, dtype=dtype)

        self.vae.requires_grad_(False)

        self.eval()

        self.final_spatial = input_image_size // 8 if not self.flatten else 1

        print("Total VAE params: ", sum(p.numel() for p in self.vae.parameters()))

        self.vae = torch.compile(self.vae, mode="reduce-overhead", fullgraph=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            latents = self.vae.encode(x.to(self.device, dtype=self.dtype)).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

            if self.flatten:
                latents = latents.flatten(start_dim=1)

            return latents


def load_sd_vae_model(
        model_name_or_path,
        input_image_size=256,
        flatten=True,
        dtype="float16",
        device="cuda",
        multi_layer=False,
    ):
    wrapper = VaeRepresentation(
        model_name_or_path,
        input_image_size=input_image_size,
        flatten=flatten,
        dtype=dtype,
        device=device,
        multi_layer=multi_layer,
    )
    return wrapper
