"""
ImageNet Latent Dataset with safetensors.
"""

import numpy as np
import torch
import json
from datasets.garment_particle_v2_dataset import GarmentParticleDatasetV2
from transformers import AutoImageProcessor, AutoTokenizer
import random
import h5py
import os
from PIL import Image
class GarmentParticleDatasetV2WithImageTextLineArt(GarmentParticleDatasetV2):
    def __init__(
        self, 
        data_dir, 
        normalization,
        text_path,
        image_dir,
        n_points=8192,
        image_processor_name="facebook/dinov2-large",
        text_tokenizer_name="openai/clip-vit-large-patch14",
        text_max_length=None,
        use_all_captions=False,
        img_drop_prob=0.1,
        text_drop_prob=0.1,
        multiplier=1.0, 
        split_file=None, 
        n_samples=None,
        fixed_n_points=True,
        pad_everything=True,
        collate_fn=None,
        front_only=False,
        front_back=False,
        **kwargs
        ):
        super().__init__(data_dir, normalization, n_points, multiplier, split_file, n_samples, fixed_n_points, collate_fn=collate_fn)
        self.image_dir = image_dir
        self.caption_dict = json.load(open(text_path, "r"))
        self.image_processor = AutoImageProcessor.from_pretrained(image_processor_name, use_fast=False)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_tokenizer_name, use_fast=False)
        self.text_max_length = text_max_length
        self.use_all_captions = use_all_captions
        self.img_drop_prob = img_drop_prob
        self.text_drop_prob = text_drop_prob
        self.pad_everything = pad_everything
        self.front_only = front_only
        self.front_back = front_back
    def __getitem__(self, idx):
        file = self.files[idx]
        name = file.split("/")[-2]
        img_paths = [os.path.join(self.image_dir, f"{name}_render_front.png"), os.path.join(self.image_dir, f"{name}_render_back.png")]
        if not self.front_back:
            if self.front_only:
                img_paths = [img_paths[0]]
            else:
                img_paths = [random.choice(img_paths)]
        imgs = [Image.open(img_path) for img_path in img_paths]
        encodings = [self.image_processor(img, return_tensors="pt") for img in imgs]
        img_tokens = []
        for encoding in encodings:
            img_token = encoding["pixel_values"]
            if random.random() < self.img_drop_prob:
                img_token = torch.zeros_like(img_token)
            img_tokens.append(img_token)
        img_tokens = torch.cat(img_tokens, dim=0)
            
        caption_list = self.caption_dict[name]
        if not self.use_all_captions:
            caption_list = random.sample(caption_list, random.randint(1, len(caption_list)))
        caption = ", ".join(caption_list)
        if random.random() < self.text_drop_prob:
            caption = ""
        encoding = self.text_tokenizer(caption, max_length=self.text_max_length, padding="max_length", truncation=True, return_tensors="pt")
        text_tokens = encoding["input_ids"].squeeze(0)
        text_attn_mask = encoding["attention_mask"].squeeze(0).bool()
        data = h5py.File(file, "r")
        pcd = self.get_pts(data)
        pcd = (pcd - np.array(self.normalization.pcd_mean)) / np.array(self.normalization.pcd_std)
        n = pcd.shape[0]
        if self.pad_everything:
            while pcd.shape[0] < self.n_points:
                pcd = np.concatenate([pcd, pcd], 0)
            pcd = pcd[:self.n_points]
        else:
            pcd = pcd[:self.n_points]
        mask = None
        if self.pad_everything:
            mask = np.ones(self.n_points).astype(bool)
            mask[n:] = False
        pcd = pcd * self.multiplier
        pcd = torch.from_numpy(pcd).float()
        
        data_dict = {
            "input": pcd,
            "pixel_values": img_tokens,
            "text_tokens": text_tokens,
            "text_attn_mask": text_attn_mask,
            "image_path": img_paths,
        }
        if mask is not None:
            data_dict["mask"] = torch.from_numpy(mask).bool()
        return data_dict
    
    def reconstruct(self, noise, mask=None, pixel_values=None, text_tokens=None, text_attn_mask=None, image_path=None, **kwargs):
        imgs, pcds, img_data = super().reconstruct(noise, mask, **kwargs)
        image_conds = []
        for paths in image_path:
            imgs = [np.array(Image.open(path)) for path in paths]
            imgs = np.concatenate(imgs, axis=1)
            image_conds.append(Image.fromarray(imgs))
        N = text_tokens.shape[0]
        captions = []
        for i in range(N):
            caption = self.text_tokenizer.decode(text_tokens[i], skip_special_tokens=True)
            captions.append(caption)
        return imgs, pcds, img_data, captions, image_conds
    
    
if __name__ == "__main__":
    from omegaconf import DictConfig
    import yaml 
    from datasets.dynamic_sampler import ConstantTokenCollator
    normalization = DictConfig(yaml.load(open("/orion/u/w4756677/garment/InteractGarment/src/configs/dataset/normalization/garment_particles_v2.2.yaml", "r"), Loader=yaml.FullLoader))
    
    from einops import rearrange
    import torch.nn.functional as F
    from flash_attn.bert_padding import index_first_axis
    def unpad_input(hidden_states, attention_mask):
        """
        Arguments:
            hidden_states: (batch, seqlen, ...)
            attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
        Return:
            hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
            indices: (total_nnz), the indices of non-masked tokens from the flattened input sequence.
            cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
            max_seqlen_in_batch: int
        """
        seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
        indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        max_seqlen_in_batch = seqlens_in_batch.detach().max().item()
        cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
        # TD [2022-03-04] We don't want to index with a bool mask, because Pytorch will expand the
        # bool mask, then call nonzero to get the indices, then index with those. The indices is @dim
        # times larger than it needs to be, wasting memory. It's faster and more memory-efficient to
        # index with integer indices. Moreover, torch's index is a bit slower than it needs to be,
        # so we write custom forward and backward to make it a bit faster.
        return (
            index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
            indices,
            cu_seqlens,
            max_seqlen_in_batch,
        )
    
    dataset = GarmentParticleDatasetV2WithImageText(
        data_dir="/orion/u/w4756677/garment/gcdv2/garment_particles_v2.1_11182025",
        gcd_list_file="/orion/u/w4756677/garment/InteractGarment/src/assets/gcd_list.txt",
        gcd_dir="/orion/u/w4756677/garment/gcdv2/garmentcodedatav2",
        image_processor_name="facebook/dinov2-large",
        img_drop_prob=0,
        normalization=normalization,
        multiplier=1.0,
        split_file="/orion/u/w4756677/garment/InteractGarment/src/assets/garment_particle_v2_train_11182025.txt",
        pad_everything=False,
    )
    collator = ConstantTokenCollator()
    from transformers import CLIPTextModel
    random_indices = np.random.randint(0, len(dataset), size=10)
    batch = [dataset[i] for i in random_indices]
    batch_dict = collator(batch)
    # batch_dict = np.load("/orion/u/w4756677/garment/InteractGarment/src/data_dict_1.npz")
    # batch_dict = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in batch_dict.items()}
    print(batch_dict.keys())
    print(batch_dict["input"].shape)
    print(batch_dict["pixel_values"].shape)
    # print(N)
    # print(batch_dict["input"])
    n_point_samples = batch_dict["cu_input_lens"][1:] - batch_dict["cu_input_lens"][:-1]
    print(n_point_samples)
    # padded_samples = torch.zeros(N, batch_dict["max_input_len"], 6)
    # mask = torch.zeros(N, batch_dict["max_input_len"], dtype=torch.bool)
    # text_embeds = text_encoder(batch_dict["text_tokens"]).last_hidden_state
    # text_embeds, indices, cu_text_lens, max_text_len = unpad_input(text_embeds, batch_dict["text_attn_mask"].bool())
    # print(len(cu_text_lens), cu_text_lens, max_text_len)
    # print(len(batch_dict["cu_input_lens"]))
    # for i in range(N):    
    #     caption = dataset.tokenizer.decode(batch_dict["text_tokens"][i], skip_special_tokens=True)
    #     print(caption)
    # for j in range(N):
    #     padded_samples[j, :n_point_samples[j]] = batch_dict["input"][batch_dict["cu_input_lens"][j]:batch_dict["cu_input_lens"][j+1]]
    #     mask[j, :n_point_samples[j]] = True
    # print(padded_samples.shape)
    # print(mask.shape)
    # imgs, pcds, img_data, captions = dataset.reconstruct(padded_samples.numpy(), mask=mask.numpy(), text_tokens=batch_dict["text_tokens"].numpy(), text_attn_mask=batch_dict["text_attn_mask"].numpy())
    # for j, (img, pcd, caption) in enumerate(zip(imgs, pcds, captions)):
    #     print(caption)
    #     img.save(f"tmp/img_{j}.png")