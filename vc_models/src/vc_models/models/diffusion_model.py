import gc
import pickle
import warnings

import deepspeed
from omegaconf.listconfig import ListConfig
import torch
import torch.nn.functional as F
from diffusers import (
    DDPMScheduler,
    AutoencoderKL,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.models.attention_processor import AttnProcessor2_0
from vc_models.models.diff_attn.trace import trace


spatial_downsampling = {
    'down_0': 16,
    'down_1': 32,
    'down_2': 64,
    'down_3': 64,
    'mid': 64,
    'up_0': 32,
    'up_1': 16,
    'up_2': 8,
    'up_3': 8,
}


def tokenize_captions(prompt, tokenizer):
    inputs = tokenizer(
        prompt,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    return inputs.input_ids


class DiffusionRepresentation(torch.nn.Module):
    def __init__(
            self,
            model_name,
            unet_path=None,
            noise_sampling="per_image",
            representation_layer_name=["mid"],
            timestep=[0],
            tokenize_captions=True,
            get_attention_maps=False,
            get_word_level_heat_map=False,
            use_cached_encoder_hidden_states=False,
            encoder_hidden_states_path=None,
            num_token_attn=8,
            input_image_size=256,
            return_text_embeddings=False,
            flatten=True,
            dtype="float16",
            device="cuda",
        ):
        super().__init__()
        self.model_name = model_name
        self.noise_sampling = noise_sampling
        self.dtype = self.get_dtype(dtype)
        self.device = device
        self.tokenize_captions = tokenize_captions
        self.flatten = flatten
        self.return_text_embeddings = return_text_embeddings
        self.get_attention_maps = get_attention_maps
        self.get_word_level_heat_map = get_word_level_heat_map
        self.num_token_attn = num_token_attn
        self.use_cached_encoder_hidden_states = use_cached_encoder_hidden_states

        if self.return_text_embeddings:
            assert not self.use_cached_encoder_hidden_states, "Cannot return text embeddings when using cached encoder hidden states for now"

        # create timesteps for each sample in the batch
        if unet_path is None:
            self.unet = UNet2DConditionModel.from_pretrained(
                model_name,
                subfolder="unet",
            ).to(device, dtype=self.dtype)
        else:
            self.unet = UNet2DConditionModel.from_pretrained(
                unet_path,
            ).to(device, dtype=self.dtype)

        if not self.use_cached_encoder_hidden_states:
            self.text_encoder = CLIPTextModel.from_pretrained(
                model_name, subfolder="text_encoder",
            ).to(device, dtype=self.dtype)
        else:
            self.text_encoder = None
            self._cached_encoder_hidden_states = pickle.load(open(encoder_hidden_states_path, "rb"))
            # convert all the values to a single tensor
            receptacle_attn_list, object_attn_list, cache_list = [], [], []
            for key in self._cached_encoder_hidden_states.keys():
                cache_list.append(torch.tensor(self._cached_encoder_hidden_states[key][0], device=device, dtype=self.dtype))
                object_attn_list.append(self._cached_encoder_hidden_states[key][1])
                receptacle_attn_list.append(self._cached_encoder_hidden_states[key][2])
            self._cached_encoder_hidden_states = torch.stack(cache_list)
            self._object_attn = torch.stack(object_attn_list).to(device=device, dtype=torch.bool)
            self._receptacle_attn = torch.stack(receptacle_attn_list).to(device=device, dtype=torch.bool)

        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_name, subfolder="tokenizer"
        )

        self.vae = AutoencoderKL.from_pretrained(
            model_name, subfolder="vae",
        ).to(device, dtype=self.dtype)

        self.noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")

        self.noise = None
        self.input_image_size = input_image_size

        # get the spatial resolution for all representation layer names
        assert type(representation_layer_name) == list or type(representation_layer_name) == ListConfig, \
            "representation_layer_name must be a list"
        self.representation_layer_name = representation_layer_name
        self.spatial_resolutions = {
            layer_name: int(input_image_size / spatial_downsampling[layer_name]) for layer_name in representation_layer_name
        } 
        self.final_spatial = max(self.spatial_resolutions.values())
        # get the block names for all representation layer names while removing duplicates
        self.block_names_for_unet = list(set([layer_name.split("_")[0] for layer_name in representation_layer_name]))

        self.timestep = torch.tensor(timestep, dtype=torch.long, device=device)
        # confirm timesteps is between 0 and num_train_timesteps
        assert torch.all((self.timestep >= 0) & (self.timestep < self.noise_scheduler.config.num_train_timesteps)), \
            f"timesteps must be between 0 and {self.noise_scheduler.config.num_train_timesteps}"

        self.unet.set_attn_processor(AttnProcessor2_0())
        if self.get_attention_maps:
            self.tc = trace(self.unet, self.tokenizer, image_size=self.input_image_size)

        ds_engine = deepspeed.init_inference(
            self.unet,
            dtype=self.dtype,
            replace_with_kernel_inject=True)
        self.unet = ds_engine.module

        ds_engine = deepspeed.init_inference(
            self.vae,
            dtype=self.dtype,
            replace_with_kernel_inject=True)
        self.vae = ds_engine.module

        self.set_timesteps(strength=1)

        self.freeze_model()


    def get_dtype(self, dtype):
        if dtype == "float16":
            return torch.float16
        elif dtype == "float32":
            return torch.float32
        elif dtype == "bfloat16":
            return torch.bfloat16


    def freeze_model(self):
        self.vae.requires_grad_(False)
        if self.text_encoder is not None:
            self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)

        self.eval()


    def encode_prompt(self, prompt):
        if self.use_cached_encoder_hidden_states:
            if prompt[0] == "":
                # raise a warning that we are using cached encoder hidden states
                warnings.warn("Prompt[0] is an empty string, using the zero index which corresponds to an empty prompt")
                prompt = [0]
            encoder_hidden_states = self._cached_encoder_hidden_states[prompt].squeeze(1)
            encoder_hidden_states = encoder_hidden_states.to(self.device)
            pooler_output = None
        else:
            if self.tokenize_captions or prompt[0] == "":
                tokens = tokenize_captions(prompt, self.tokenizer).to(self.device)
            else:
                tokens = prompt.to(self.device)
            clip_output = self.text_encoder(tokens)
            encoder_hidden_states = clip_output.last_hidden_state
            pooler_output = clip_output.pooler_output
            
        return encoder_hidden_states, pooler_output


    def forward(self, x, prompt=[""], search_word=None, sample_noise=True):
        assert x.max() <= 1.0 and x.min() >= -1.0, "Input image must be between -1 and 1"

        if self.get_attention_maps:
            search_word = search_word if search_word is not None else prompt
            with self.tc as tc:
                representation = self._get_representation(x, prompt, sample_noise)
                select_prompts = []
                if self.use_cached_encoder_hidden_states:
                    if prompt[0] == "":
                        # raise a warning that we are using cached encoder hidden states
                        warnings.warn("Prompt[0] is an empty string, using a random prompt")
                        prompt = [0]
                    
                    if self.num_token_attn == 2:
                        # hardcoded for now
                        select_prompts = (
                            self._object_attn[prompt].squeeze(1),
                            self._receptacle_attn[prompt].squeeze(1)
                        )
                    elif self.num_token_attn == 1:
                        select_prompts = (
                            self._object_attn[prompt].squeeze(1),
                        )

                heat_map = tc.compute_global_heat_map(
                    prompt=prompt,
                    get_global_heat_map=self.get_word_level_heat_map,
                    num_tokens=self.num_token_attn,
                    select_prompts=select_prompts,
                )
                if self.get_word_level_heat_map:
                    heat_map = heat_map.compute_word_heat_map(search_word)
                return representation, heat_map
        else:
            return self._get_representation(x, prompt, sample_noise)


    def set_timesteps(self, strength):
        max_timestep = max(self.timestep).item()

        total_steps = self.noise_scheduler.config.num_train_timesteps
        num_inference_steps = 50
        denoising_start = 1 - (max_timestep + 1)/ total_steps

        # get the original timestep using init_timestep
        if denoising_start is None:
            init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
            t_start = max(num_inference_steps - init_timestep, 0)
        else:
            t_start = 0
        
        timesteps = self.noise_scheduler.timesteps[t_start * self.noise_scheduler.order :]

        if denoising_start is not None:
            discrete_timestep_cutoff = int(
                round(total_steps - (denoising_start * total_steps))
            )
            timesteps = list(filter(lambda ts: ts < discrete_timestep_cutoff, timesteps))
            timesteps = torch.tensor(timesteps)

        timesteps = timesteps[::int(total_steps/num_inference_steps)]
        self.noise_scheduler.set_timesteps(num_inference_steps, device='cuda')


    def _get_representation(self, x, prompt=[""], sample_noise=True):
        t = self.timestep

        with torch.inference_mode():
            # TODO: Decide between sample and mean
            latents = self.vae.encode(x.to(self.device, dtype=self.dtype)).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            if self.noise_sampling == "per_image":
                noise = torch.randn_like(latents)
            elif self.noise_sampling == "per_episode":
                # When we sample noise per episode, we need to depend on an external 
                # variable to know if its time to sample a new noise
                if sample_noise or self.noise is None:
                    self.noise = torch.randn_like(latents[0])
                    # repeat the noise for each sample in the batch
                noise = self.noise.repeat(latents.shape[0], 1, 1, 1)
            elif self.noise_sampling == "per_run":
                # When we sample noise per run, we just use the same noise for all episodes
                if self.noise is None:
                    self.noise = torch.randn_like(latents[0])
                # repeat the noise for each sample in the batch
                noise = self.noise.repeat(latents.shape[0], 1, 1, 1)
            else:
                raise ValueError(f"Invalid noise sampling method: {self.noise_sampling}")

            bsz = latents.shape[0]

            timesteps = t.repeat(bsz, 1).to(self.device)

            encoder_hidden_states, pooler_output = self.encode_prompt(prompt)

            if encoder_hidden_states.shape[0] == 1:
                # If only one caption was passed in, repeat it for each sample in the batch
                encoder_hidden_states = encoder_hidden_states.repeat(bsz, 1, 1)

            model_pred = []
            # TODO: get rid of for loop
            for i in range(timesteps.shape[1]):
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps[:, i])
                # average out the spatial dimensions and append
                layer_dict = self.unet(
                    noisy_latents,
                    timesteps[:, i],
                    encoder_hidden_states,
                    get_block_representations=self.block_names_for_unet,
                )

                for layer_name in self.representation_layer_name:
                    result = layer_dict[layer_name]

                    if result.shape[-1] != self.final_spatial:
                        result = F.interpolate(
                            result, size=(self.final_spatial, self.final_spatial), mode="bilinear"
                        )

                    if self.flatten:
                        result = result.flatten(start_dim=1)

                    model_pred.append(result)

            output = torch.cat(model_pred, dim=1).detach()

        if self.return_text_embeddings:
            return output.clone(), pooler_output
        else:
            return output.clone()


def load_sd_model(
        model_name,
        unet_path=None,
        noise_sampling="per_image",
        representation_layer_name=["mid"],
        timestep=[199],
        tokenize_captions=True,
        get_attention_maps=False,
        get_word_level_heat_map=False,
        use_cached_encoder_hidden_states=False,
        encoder_hidden_states_path=None,
        num_token_attn=8,
        input_image_size=256,
        return_text_embeddings=False,
        flatten=True,
        dtype="float16",
        device="cuda"
    ):
    wrapper = DiffusionRepresentation(
        model_name,
        unet_path=unet_path,
        noise_sampling=noise_sampling,
        representation_layer_name=representation_layer_name,
        timestep=timestep,
        tokenize_captions=tokenize_captions,
        get_attention_maps=get_attention_maps,
        get_word_level_heat_map=get_word_level_heat_map,
        use_cached_encoder_hidden_states=use_cached_encoder_hidden_states,
        encoder_hidden_states_path=encoder_hidden_states_path,
        num_token_attn=num_token_attn,
        input_image_size=input_image_size,
        return_text_embeddings=return_text_embeddings,
        flatten=flatten,
        dtype=dtype,
        device=device
    )
    return wrapper
