import numpy as np
import torch as th
import torch.nn.functional as F
import torch.nn as nn
from torchdiffeq import odeint
from functools import partial
from tqdm import tqdm
from .utils import _sample_tokens


class sde:
    """SDE solver class"""
    def __init__(
        self, 
        drift,
        diffusion,
        *,
        t0,
        t1,
        num_steps,
        sampler_type,
    ):
        assert t0 < t1, "SDE sampler has to be in forward time"

        self.num_timesteps = num_steps
        self.t = th.linspace(t0, t1, num_steps)
        self.dt = self.t[1] - self.t[0]
        self.drift = drift
        self.diffusion = diffusion
        self.sampler_type = sampler_type

    def __Euler_Maruyama_step(self, x, mean_x, t, model, **model_kwargs):
        w_cur = th.randn(x.size()).to(x)
        t = th.ones(x.size(0)).to(x) * t
        dw = w_cur * th.sqrt(self.dt)
        drift = self.drift(x, t, model, **model_kwargs)
        diffusion = self.diffusion(x, t)
        mean_x = x + drift * self.dt
        x = mean_x + th.sqrt(2 * diffusion) * dw
        return x, mean_x
    
    def __Heun_step(self, x, _, t, model, **model_kwargs):
        w_cur = th.randn(x.size()).to(x)
        dw = w_cur * th.sqrt(self.dt)
        t_cur = th.ones(x.size(0)).to(x) * t
        diffusion = self.diffusion(x, t_cur)
        xhat = x + th.sqrt(2 * diffusion) * dw
        K1 = self.drift(xhat, t_cur, model, **model_kwargs)
        xp = xhat + self.dt * K1
        K2 = self.drift(xp, t_cur + self.dt, model, **model_kwargs)
        return xhat + 0.5 * self.dt * (K1 + K2), xhat # at last time point we do not perform the heun step

    def __forward_fn(self):
        """TODO: generalize here by adding all private functions ending with steps to it"""
        sampler_dict = {
            "Euler": self.__Euler_Maruyama_step,
            "Heun": self.__Heun_step,
        }

        try:
            sampler = sampler_dict[self.sampler_type]
        except:
            raise NotImplementedError("Smapler type not implemented.")
    
        return sampler

    def sample(self, init, model, **model_kwargs):
        """forward loop of sde"""
        x = init
        mean_x = init 
        samples = []
        sampler = self.__forward_fn()
        for ti in self.t[:-1]:
            with th.no_grad():
                x, mean_x = sampler(x, mean_x, ti, model, **model_kwargs)
                samples.append(x)

        return samples

class ode:
    """ODE solver class"""
    def __init__(
        self,
        drift,
        *,
        t0,
        t1,
        sampler_type,
        num_steps,
        atol,
        rtol,
        timestep_shift,
        curve_sampling=False,
        stitch_sampling=False,
    ):
        self.drift = drift
        self.t = th.linspace(t0, t1, num_steps)
        self.curve_sampling = curve_sampling
        self.stitch_sampling = stitch_sampling

        if timestep_shift > 0:
            def compute_tm(t_n, timestep_shift):
                numerator = timestep_shift * t_n
                denominator = 1 + (timestep_shift - 1) * t_n
                return numerator / denominator
            self.t = th.tensor([compute_tm(t_n, timestep_shift) for t_n in self.t])

        self.atol = atol
        self.rtol = rtol
        self.sampler_type = sampler_type

    def sample(self, x, model, logger=None, **model_kwargs):
        device = x[0].device if isinstance(x, tuple) else x.device
        def _fn(t, x):
            if logger is not None:
                logger.info(f"T={t.cpu().item()}")
            t = th.ones(x[0].size(0)).to(device) * t if isinstance(x, tuple) else th.ones(x.size(0)).to(device) * t
            model_output = self.drift(x, t, model, **model_kwargs)
            return model_output["pred"]

        t = self.t.to(device)
        atol = [self.atol] * len(x) if isinstance(x, tuple) else [self.atol]
        rtol = [self.rtol] * len(x) if isinstance(x, tuple) else [self.rtol]
        samples = odeint(
            _fn,
            x,
            t,
            method=self.sampler_type,
            atol=atol,
            rtol=rtol
        )
        return samples
    
    def sample_euler(self, x, model, **model_kwargs):
        device = x[0].device if isinstance(x, tuple) else x.device
        xs = [x]
        pbar = tqdm(range(len(self.t[:-1])), total=len(self.t[:-1]))
        pbar.set_description("Sampling")
        for i, (t1, t2) in enumerate(zip(self.t[:-1], self.t[1:])):
            pbar.update(1)
            pbar.set_postfix(t1=t1.item(), t2=t2.item())
            dt = t2 - t1
            t1 = th.ones(x[0].size(0)).to(device) * t1 if isinstance(x, tuple) else th.ones(x.size(0)).to(device) * t1
            t2 = th.ones(x[0].size(0)).to(device) * t2 if isinstance(x, tuple) else th.ones(x.size(0)).to(device) * t2
            model_output = self.drift(x, t1, model, **model_kwargs) # v_t
            drift = model_output["pred"]
            x = x + drift * dt
            xs.append(x)
        return {
            "preds": xs,
        }
            
    def inversion_fixed_point_sampling(self, x, model, **model_kwargs):
        """https://arxiv.org/pdf/2411.15843"""
        device = x[0].device if isinstance(x, tuple) else x.device
        ts = th.flip(self.t, dims=(0,))
        xs = [x]
        for t1, t2 in zip(ts[:-1], ts[1:]): # t2 < t1
            _xts = []
            xtm1 = x.clone()
            _t1 = th.ones(x[0].size(0)).to(device) * t1 if isinstance(x, tuple) else th.ones(x.size(0)).to(device) * t1
            for i in range(4):
                drift = self.drift(xtm1, _t1, model, **model_kwargs)
                xtm1 = x + drift * (t2 - t1)
                _xts.append(xtm1)
            x = th.mean(th.stack(_xts), dim=0)
            xs.append(x)
        return xs
    
    def guided_sampling_flowchef(self, x, model, loss_fn, opt_steps, guide_ratio, **model_kwargs):
        # Algorithm from https://arxiv.org/pdf/2412.00100
        device = x[0].device if isinstance(x, tuple) else x.device

        diff = self.t[1:] - self.t[:-1]
        early_stop = int(len(diff) * guide_ratio)
        step_count = 0
        xs = [x]
        for t, dt in zip(self.t[:-1], diff):
            t = th.ones(x[0].size(0)).to(device) * t if isinstance(x, tuple) else th.ones(x.size(0)).to(device) * t
            # optimization step
            if step_count < early_stop:
                with th.enable_grad():
                    x.requires_grad_(True)
                    optimizer = th.optim.Adam([x], lr=1e-2)
                    for n in range(opt_steps):
                        drift = self.drift(x, t, model, **model_kwargs) # v_t
                        x_1_t = x + drift * (1 - t) # x_1 given x_t
                        loss_dict = loss_fn(x_1_t, t)
                        optimizer.zero_grad()
                        loss_dict["total_loss"].backward()
                        optimizer.step()
                        s = f"t: {t.item()}, step: {n},"
                        for k, v in loss_dict.items():
                            s += f"{k}: {v.item()}, "
                        print(s)
                step_count += 1
            else:
                drift = self.drift(x, t, model, **model_kwargs) # v_t
            # euler step
            x = x + drift * dt
            xs.append(x)
        return xs, self.t
    
    def guided_sampling_flowdps(self, x, model, loss_fn, opt_steps, guide_ratio, **model_kwargs):
        # Algorithm from https://arxiv.org/pdf/2503.08136
        device = x[0].device if isinstance(x, tuple) else x.device

        early_stop = int(len(self.t[:-1]) * guide_ratio)
        step_count = 0
        xs = [x]
        for t1, t2 in zip(self.t[:-1], self.t[1:]):
            t1 = th.ones(x[0].size(0)).to(device) * t1 if isinstance(x, tuple) else th.ones(x.size(0)).to(device) * t1
            t2 = th.ones(x[0].size(0)).to(device) * t2 if isinstance(x, tuple) else th.ones(x.size(0)).to(device) * t2
            drift = self.drift(x, t1, model, **model_kwargs) # v_t
            x_1_t = x + drift * (1 - t1) # x_1 given x_t
            x_0_t = x - drift * t1
            if step_count < early_stop:
                x_1_t_hat = x_1_t.clone()
                # optimization step
                with th.enable_grad():
                    x_1_t_hat.requires_grad_(True)
                    optimizer = th.optim.Adam([x_1_t_hat], lr=1)
                    for n in range(opt_steps):
                        loss_dict = loss_fn(x_1_t_hat, t1)
                        optimizer.zero_grad()
                        loss_dict["total_loss"].backward()
                        optimizer.step()
                        s = f"t: {t1.item()}, step: {n},"
                        for k, v in loss_dict.items():
                            s += f"{k}: {v.item() if isinstance(v, th.Tensor) else v}, "
                        print(s)
                x_1_t = (1 - t1) * x_1_t_hat + t1 * x_1_t
                step_count += 1
            # eps = th.randn_like(x_0_t)
            # x_0_t = th.sqrt(1 - t2) * x_0_t + th.sqrt(t2) * eps
            loss_dict = loss_fn(x_1_t, t1)
            s = f"t: {t1.item()} final, "
            for k, v in loss_dict.items():
                s += f"{k}: {v.item() if isinstance(v, th.Tensor) else v}, "
            print(s)
            x = t2 * x_1_t + (1 - t2) * x_0_t
            xs.append(x)
        return xs, self.t
    