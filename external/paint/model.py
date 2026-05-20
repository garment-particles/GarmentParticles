import hydra
import torch
from transport import create_transport, Sampler

class Model:
    def __init__(self, cfg, checkpoint_path):
        # Create model:
        self.model = hydra.utils.instantiate(cfg.model)
        for p in self.model.parameters():
            p.requires_grad = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        transport = hydra.utils.instantiate(cfg.transport)
        sampler = Sampler(transport)
        self.sample_fn = sampler.sample_ode(
            sampling_method=cfg.sample.sampling_method,
            num_steps=cfg.sample.num_sampling_steps,
            atol=cfg.sample.atol,
            rtol=cfg.sample.rtol,
            reverse=cfg.sample.reverse,
            timestep_shift=cfg.sample.timestep_shift,
            curve_sampling=cfg.sample.curve_sampling,
            stitch_sampling=cfg.sample.stitch_sampling
        )
        self.model.eval()
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage, weights_only=False)
        model_dict = checkpoint['ema']
        model_dict = {k.replace('module.', ''): v for k, v in model_dict.items()}
        self.model.load_state_dict(model_dict)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.dataset.tokenizer_name, use_fast=False)
        self.pcd_mean = np.array(cfg.dataset.pcd_mean)
        self.pcd_std = np.array(cfg.dataset.pcd_std)
        self.xlims = np.array(cfg.dataset.xlims)
        self.ylims = np.array(cfg.dataset.ylims)
    @torch.no_grad()
    def sample(self, prompt=None):
        z = torch.randn(1, 6000, 6).to("cuda")
        prompt = "" if prompt is None else prompt
        encoding = self.tokenizer(prompt, return_tensors="pt")
        text_attn_mask = encoding["attention_mask"]
        data_dict["text_tokens"] = encoding["input_ids"]
        data_dict["text_attn_mask"] = encoding["attention_mask"]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = self.sample_fn(z, self.model.forward, **data_dict)
        preds = output["preds"][-1].cpu().numpy()
        preds = (preds * self.pcd_std) + self.pcd_mean
        return preds