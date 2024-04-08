from typing import List, Type
import math

from diffusers.models.attention_processor import Attention
import numpy as np
import PIL.Image as Image
import torch
import torch.nn.functional as F
import ipdb
from .utils import auto_autocast
from .heatmap import RawHeatMapCollection, GlobalHeatMap, SimpleHeatMapCollection
from .hook import ObjectHooker, AggregateHooker, UNetCrossAttentionLocator

__all__ = ['trace', 'DiffusionHeatMapHooker', 'GlobalHeatMap']


class DiffusionHeatMapHooker(AggregateHooker):
    def __init__(
            self,
            unet,
            tokenizer,
            image_size=512,
    ):
        latent_h = image_size // 8
        self.latent_hw = latent_h ** 2

        self.all_heat_maps = SimpleHeatMapCollection(latent_h, latent_h)
        self.locator = UNetCrossAttentionLocator(restrict=None, locate_middle_block=True)

        modules = [
            UNetCrossAttentionHooker(
                x,
                self,
                layer_idx=idx,
                latent_hw=self.latent_hw,
            ) for idx, x in enumerate(self.locator.locate(unet))
        ]

        super().__init__(modules)
        self.tokenizer = tokenizer

    @property
    def layer_names(self):
        return self.locator.layer_names

    def compute_global_heat_map(
            self, 
            prompt=None,
            normalize=False,
            get_global_heat_map=True,
            num_tokens=8,
            select_prompts=[],
        ):
        # type: (str, List[float], int, int, bool) -> GlobalHeatMap
        """
        Compute the global heat map for the given prompt, aggregating across time (inference steps) and space (different
        spatial transformer block heat maps).

        Args:
            prompt: The prompt to compute the heat map for. If none, uses the last prompt that was used for generation.

        Returns:
            A heat map object for computing word-level heat maps.
        """
        heat_maps = self.all_heat_maps
        if prompt is None:
            raise ValueError('Prompt must be specified.')

        with auto_autocast(dtype=torch.float32):
            maps = heat_maps.get_avg_heatmap().clone()

        heat_maps.clear()
        if get_global_heat_map:
            maps = maps[:, :num_tokens + 2]  # 1 for SOS and 1 for padding

            if normalize:
                maps = maps / (maps[1:-1].sum(0, keepdim=True) + 1e-6)  # drop out [SOS] and [PAD] for proper probabilities
            return GlobalHeatMap(self.tokenizer, prompt, maps.squeeze(), self.maps_bkp)
        else:
            if len(select_prompts) != 0:
                maps_list = []
                for select_prompt in select_prompts:
                    maps_list.append((maps * select_prompt.unsqueeze(-1).unsqueeze(-1)).sum(1))
                maps = torch.stack(maps_list, dim=1)
            else:
                maps = maps[:, 1:num_tokens + 1]
            if normalize:
                maps = maps / (maps.sum(0, keepdim=True) + 1e-6)
            return maps

def tensor_to_pil(tensor: torch.Tensor):
    tensor = tensor.squeeze().cpu().numpy().astype(np.uint8)
    # permute to (batch, height, width, channels)
    tensor = np.transpose(tensor, (1, 2, 0))
    return Image.fromarray(tensor)


class UNetCrossAttentionHooker(ObjectHooker[Attention]):
    def __init__(
            self,
            module: Attention,
            parent_trace: 'trace',
            context_size: int = 77,
            layer_idx: int = 0,
            latent_hw: int = 9216,

    ):
        super().__init__(module)
        self.heat_maps = parent_trace.all_heat_maps
        self.context_size = context_size
        self.layer_idx = layer_idx
        self.latent_hw = latent_hw
        self.trace = parent_trace


    @torch.no_grad()
    def _unravel_attn(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        # x shape: (heads, height * width, tokens)
        """
        Unravels the attention, returning it as a collection of heat maps.

        Args:
            x (`torch.Tensor`): cross attention slice/map between the words and the tokens.
            value (`torch.Tensor`): the value tensor.

        Returns:
            `List[Tuple[int, torch.Tensor]]`: the list of heat maps across heads.
        """
        h = w = int(math.sqrt(x.size(1)))
        maps = []
        x = x.permute(2, 0, 1)

        with auto_autocast(dtype=torch.float32):
            for map_ in x:
                map_ = map_.view(map_.size(0), h, w)
                # map_ = map_[map_.size(0) // 2:]  # Filter out unconditional
                maps.append(map_)

        maps = torch.stack(maps, 0)  # shape: (tokens, heads, height, width)
        return maps.permute(1, 0, 2, 3).contiguous()  # shape: (heads, tokens, height, width)

    def __call__(
            self,
            attn: Attention,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
    ):
        """Capture attentions and aggregate them."""
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross is not None:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # compute shape factor
        factor = int(math.sqrt(self.latent_hw // attention_probs.shape[1]))

        # skip if too large
        if attention_probs.shape[-1] == self.context_size and factor != 8:
            # shape: (batch_size, 64 // factor, 64 // factor, 77)
            maps = self._unravel_attn(attention_probs)
            maps = self.batch_to_head_dim(attn.heads, maps)

            num_heads = maps.shape[1]

            self.heat_maps.update(maps.sum(1), num_heads)
            # for head_idx in range(len(maps[0])):
                # self.heat_maps.update(factor, self.layer_idx, head_idx, maps[:, head_idx])

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

    def batch_to_head_dim(self, head_size, tensor):
        batch_size, seq_len, dimx, dimy = tensor.shape
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dimx, dimy)
        return tensor

    def _hook_impl(self):
        self.module.set_processor(self)

    @property
    def num_heat_maps(self):
        return self.heat_maps.num_maps
        # return len(next(iter(self.heat_maps.values())))


trace: Type[DiffusionHeatMapHooker] = DiffusionHeatMapHooker
