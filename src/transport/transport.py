import torch as th
import torch.nn.functional as F
import numpy as np
import logging
import math
import random
from typing import Optional, Dict, Any, Tuple, List

import enum

from . import path
from .utils import EasyDict, log_state, mean_flat
from .integrators import ode, sde
from scipy.stats import norm

import contextlib
from omegaconf import DictConfig, OmegaConf


def unwrap_model_output(out: Any) -> th.Tensor:
    if isinstance(out, dict):
        if "pred" in out:
            return out["pred"]
        for v in out.values():
            if th.is_tensor(v):
                return v
        raise RuntimeError("Model returned dict but no tensor found.")
    if not th.is_tensor(out):
        raise RuntimeError(f"Model output must be Tensor/dict, got {type(out)}")
    return out


def ddpo_transport_step_with_logprob(
    transport,
    *,
    model_output: th.Tensor,     # [N,...] (velocity)
    t_cur: th.Tensor,            # [N]
    t_next: th.Tensor,           # [N]
    sample: th.Tensor,           # x_t [N,...]
    noise_level: float,
    prev_sample: Optional[th.Tensor] = None,  # if provided => compute logprob at this x_{t+1}
    sigma_max_fallback: Optional[float] = None,
    sigma_clip_eps: float = 1e-5,
):
    """
    DDPO 'sde' math, but using transport.path_sampler.compute_sigma_t(t).
    Numerical fixes:
    - sigma/logprob path forced fp32 (call-site should disable autocast)
    - optional sigma clamp to avoid denom->0 blow-ups
    - strict finite checks (optional)
    """
    # fp32 safety
    model_output = model_output.float()
    sample_f = sample.float()
    t_cur_f = t_cur.float()
    t_next_f = t_next.float()
    if prev_sample is not None:
        prev_sample = prev_sample.float()


    # IMPORTANT: compute_sigma_t should be executed in fp32 WITHOUT autocast
    sigma_cur, _ = transport.path_sampler.compute_sigma_t(path.expand_t_like_x(t_cur_f, sample_f))
    sigma_next, _ = transport.path_sampler.compute_sigma_t(path.expand_t_like_x(t_next_f, sample_f))
    # sigma = 1 - t 
    # t = 0: noisy; t = 1: clean
    # t_cur < t_next
    # sigma_cur > sigma_next


    # sigma_cur = path.expand_t_like_x(t_cur_f, sample_f)
    # sigma_next = path.expand_t_like_x(t_next_f, sample_f)

    # Safety: keep sigma in a sane open interval for DDPO math (debuggable)
    # If your theory requires sigma outside (0,1), you must adjust DDPO math accordingly.
    sigma_cur = sigma_cur.float().clamp(min=sigma_clip_eps, max=1.0 - sigma_clip_eps)
    sigma_next = sigma_next.float().clamp(min=sigma_clip_eps, max=1.0 - sigma_clip_eps)

    # When noise_level=0, use t-space dt directly to match pure ODE behavior
    # This avoids numerical differences from sigma clamp operations
    # if noise_level == 0.0:
    #     dt = t_next_f - t_cur_f  # Use t-space dt for exact ODE equivalence
    # else:
    #     dt = -sigma_next + sigma_cur  # Use sigma-space dt for SDE 

  
    dt = -sigma_next + sigma_cur  # Use sigma-space dt for SDE 

    # dt_true = (t_next_f - t_cur_f).unsqueeze(-1)

    # print('dt_true d_dt', dt_true, (dt_true - dt))

    # dt = (t_next_f - t_cur_f).unsqueeze(-1)


    sigma_max = th.as_tensor(1.0 - sigma_clip_eps, device=sample_f.device, dtype=th.float32)

    denom = (1.0 - sigma_cur).clamp(min=sigma_clip_eps)
    denom = denom.clamp(min=1e-12)
    std_dev_t = th.sqrt((sigma_cur / denom).clamp(min=1e-12)) * float(noise_level) 

    sigma_safe = sigma_cur.clamp(min=1e-12)
    std2 = std_dev_t * std_dev_t

   

    prev_mean = (
        sample_f * (1.0 - std2 / (2.0 * sigma_safe) * dt)
        + model_output * (1.0 + std2 * (1.0 - sigma_cur) / (2.0 * sigma_safe)) * dt
        # + model_output * (1.0 + std2 * (1.0 - sigma_cur) / (2.0 * sigma_safe)) * (t_next_f - t_cur_f).unsqueeze(-1)
    )

    # pev_mean = (
    #     sample_f + model_output * dt
    #     # + model_output * (1.0 + std2 * (1.0 - sigma_cur) / (2.0 * sigma_safe)) * (t_next_f - t_cur_f)
    # )

    neg_dt = (dt).clamp(min=1e-12)
    sqrt_neg_dt = th.sqrt(neg_dt)

    if prev_sample is None:
        # th.manual_seed(42)
        # th.cuda.manual_seed_all(42)
        eps = th.randn_like(sample_f)
        prev_sample = prev_mean + std_dev_t * sqrt_neg_dt * eps


    scale = (std_dev_t * sqrt_neg_dt).clamp(min=1e-12)
    log_sqrt_2pi = 0.5 * math.log(2.0 * math.pi)
    # print('scale', scale)

    # logprob should NOT backprop through x_next; detach prev_sample
    log_prob_elem = (
        -((prev_sample.detach() - prev_mean) ** 2) / (2.0 * (scale ** 2))
        - th.log(scale)
        - log_sqrt_2pi
    )
    log_prob = log_prob_elem.mean(dim=tuple(range(1, log_prob_elem.ndim)))  # [N]



    return prev_sample, log_prob, prev_mean, std_dev_t


def _get_pred(x):
    return x.get("pred", x) if isinstance(x, dict) else x

def _as_fp32(x: th.Tensor) -> th.Tensor:
    return x.float() if x.dtype != th.float32 else x


class ModelType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    NOISE = enum.auto()  # the model predicts epsilon
    SCORE = enum.auto()  # the model predicts \nabla \log p(x)
    VELOCITY = enum.auto()  # the model predicts v(x)
    FFD = enum.auto()  # the model predicts f(x)

class PathType(enum.Enum):
    """
    Which type of path to use.
    """

    LINEAR = enum.auto()
    GVP = enum.auto()
    VP = enum.auto()
class WeightType(enum.Enum):
    """
    Which type of weighting to use.
    """

    NONE = enum.auto()
    VELOCITY = enum.auto()
    LIKELIHOOD = enum.auto()


class Transport:

    def __init__(
        self,
        *,
        model_type,
        path_type,
        loss_type,
        train_eps,
        sample_eps,
        use_lognorm=False,
    ):
        
        if model_type == "noise":
            model_type = ModelType.NOISE
        elif model_type == "score":
            model_type = ModelType.SCORE
        elif model_type == "ffd":
            model_type = ModelType.FFD
        else:
            model_type = ModelType.VELOCITY

        if loss_type == "velocity":
            loss_type = WeightType.VELOCITY
        elif loss_type == "likelihood":
            loss_type = WeightType.LIKELIHOOD
        else:
            loss_type = WeightType.NONE

        path_choice = {
            "Linear": PathType.LINEAR,
            "GVP": PathType.GVP,
            "VP": PathType.VP,
        }

        path_type = path_choice[path_type]

        if (path_type in [PathType.VP]):
            train_eps = 1e-5 if train_eps is None else train_eps
            sample_eps = 1e-3 if train_eps is None else sample_eps
        elif (path_type in [PathType.GVP, PathType.LINEAR] and model_type != ModelType.VELOCITY):
            train_eps = 1e-3 if train_eps is None else train_eps
            sample_eps = 1e-3 if train_eps is None else sample_eps
        else: # velocity & [GVP, LINEAR] is stable everywhere
            train_eps = 0
            sample_eps = 0
            
            
        path_options = {
            PathType.LINEAR: path.ICPlan,
            PathType.GVP: path.GVPCPlan,
            PathType.VP: path.VPCPlan,
        }

        self.loss_type = loss_type
        self.model_type = model_type
        self.path_sampler = path_options[path_type]()
        self.train_eps = train_eps
        self.sample_eps = sample_eps
        self.use_lognorm = use_lognorm
    def prior_logp(self, z):
        '''
            Standard multivariate normal prior
            Assume z is batched
        '''
        shape = th.tensor(z.size())
        N = th.prod(shape[1:])
        _fn = lambda x: -N / 2. * np.log(2 * np.pi) - th.sum(x ** 2) / 2.
        return th.vmap(_fn)(z)
    

    def check_interval(
        self, 
        train_eps, 
        sample_eps, 
        *, 
        diffusion_form="SBDM",
        sde=False, 
        reverse=False, 
        eval=False,
        last_step_size=0.0,
    ):
        t0 = 0
        t1 = 1
        eps = train_eps if not eval else sample_eps
        if (type(self.path_sampler) in [path.VPCPlan]):

            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        elif (type(self.path_sampler) in [path.ICPlan, path.GVPCPlan]) \
            and (self.model_type != ModelType.VELOCITY or sde): # avoid numerical issue by taking a first semi-implicit step

            t0 = eps if (diffusion_form == "SBDM" and sde) or self.model_type != ModelType.VELOCITY else 0
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size
        else:
            t0 = eps 
            t1 = 1 - eps
        
        if reverse:
            t0, t1 = 1 - t0, 1 - t1

        return t0, t1

    def sample_logit_normal(self, mu, sigma, size=1):
        # Generate samples from the normal distribution
        samples = norm.rvs(loc=mu, scale=sigma, size=size)
        
        # Transform samples to be in the range (0, 1) using the logistic function
        samples = 1 / (1 + np.exp(-samples))

        # Numpy to Tensor
        samples = th.tensor(samples, dtype=th.float32)

        return samples

    def sample_in_range(self, mu, sigma, target_size, range_min=0, range_max=0.5):
        samples = []
        while len(samples) < target_size:
            generated_samples = self.sample_logit_normal(mu, sigma, size=target_size)
            filtered_samples = generated_samples[(generated_samples >= range_min) & (generated_samples <= range_max)]
            samples.extend(filtered_samples)
        
        # If we have more than the target size, truncate the list
        samples = samples[:target_size]
        return th.tensor(samples)

    def sample(self, x1, sp_timesteps=None, shifted_mu=0, batch_size=None):
        """Sampling x0 & t based on shape of x1 (if needed)
          Args:
            x1 - data point; [batch, *dim]
        """
        if self.model_type == ModelType.FFD:
            return None, None, x1
        
        if batch_size is None:
            batch_size = x1.shape[0]
        x0 = th.randn_like(x1)
        t0, t1 = self.check_interval(self.train_eps, self.sample_eps)
        if not self.use_lognorm:
            t = th.rand((batch_size,)) * (t1 - t0) + t0
        else:
            t = self.sample_logit_normal(shifted_mu, 1, size=batch_size) * (t1 - t0) + t0
        
        # overwrite t if sp_timesteps is provided (for validation)
        if sp_timesteps is not None:
            # uniform sampling between self.sp_timesteps[0] and self.sp_timesteps[1]
            t = th.rand((batch_size,)) * (sp_timesteps[1] - sp_timesteps[0]) + sp_timesteps[0]

        t = t.to(x1)
        return t, x0, x1
    

    def training_losses(
        self, 
        model,  
        x1, 
        model_kwargs=None,
        sp_timesteps=None,
        shifted_mu=0,
        return_pred=False,
    ):
        """Loss for training the score model
        Args:
        - model: backbone model; could be score, noise, or velocity
        - x1: datapoint
        - model_kwargs: additional arguments for the model
        """
        if model_kwargs == None:
            model_kwargs = {}
        mask = model_kwargs.get('mask', None)
        B = model_kwargs.get('batch_size', None)
        cu_input_lens = model_kwargs.get('cu_input_lens', None)
        if B is None:
            B = x1.shape[0]
        if mask is not None:
            mask = mask[..., None].expand_as(x1)
        if self.model_type != ModelType.FFD:
            
            t, x0, x1 = self.sample(x1, sp_timesteps, shifted_mu, batch_size=B)
            if cu_input_lens is not None:
                # example: t = [0, 1, 2], cu_input_lens = [0, 1, 5, 10]
                # then I want to make t = [0, 1, 1, 1, 1, 2, 2, 2, 2, 2] so 0 is repeated 1 time, 1 is repeated 4 times, 2 is repeated 5 times
                lens = cu_input_lens[1:] - cu_input_lens[:-1]
                t = t.repeat_interleave(lens.int(), dim=0)
            t, xt, ut = self.path_sampler.plan(t, x0, x1)

        else:
            t = None
            xt = x1
            ut = x1
        
        model_output = model(xt, t=t, **model_kwargs)
        model_output = model_output.get("pred", model_output)
        

        terms = {}
        if return_pred:
            terms['pred'] = model_output
        if self.model_type == ModelType.VELOCITY:
            loss = ((model_output - ut) ** 2)
            if cu_input_lens is not None:
                # 1. Average over channels -> Shape (Total_Tokens,)
                token_loss = loss.mean(-1) 
                loss = th.stack([token_loss[cu_input_lens[i]:cu_input_lens[i+1]].mean() for i in range(len(cu_input_lens) - 1)])

            else:
                loss = mean_flat(((model_output - ut) ** 2), mask=mask)
            terms['mse_loss'] = loss
        elif self.model_type == ModelType.FFD:
            loss = ((model_output - ut) ** 2)
            loss = mean_flat(((model_output - ut) ** 2), mask=mask)
            terms['mse_loss'] = loss
        else: 
            _, drift_var = self.path_sampler.compute_drift(xt, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, xt))
            if self.loss_type in [WeightType.VELOCITY]:
                weight = (drift_var / sigma_t) ** 2
            elif self.loss_type in [WeightType.LIKELIHOOD]:
                weight = drift_var / (sigma_t ** 2)
            elif self.loss_type in [WeightType.NONE]:
                weight = 1
            else:
                raise NotImplementedError()
            
            if self.model_type == ModelType.NOISE:
                terms['mse_loss'] = mean_flat(weight * ((model_output - x0) ** 2), mask=mask)
            else:
                terms['mse_loss'] = mean_flat(weight * ((model_output * sigma_t + x0) ** 2), mask=mask)
                
                
        return terms
    

    def get_drift(
        self
    ):
        """member function for obtaining the drift of the probability flow ODE"""
        def score_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            model_output = model(x, t=t, **model_kwargs)
            return (-drift_mean + drift_var * model_output) # by change of variable
        
        def noise_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))
            model_output = model(x, t=t, **model_kwargs)
            score = model_output / -sigma_t
            return (-drift_mean + drift_var * score)
        
        def velocity_ode(x, t, model, **model_kwargs):
            model_output = model(x, t=t, **model_kwargs)
            return model_output

        def ffd_ode(x, t, model, **model_kwargs):
            model_output = model(x, **model_kwargs)
            return model_output

        if self.model_type == ModelType.NOISE:
            drift_fn = noise_ode
        elif self.model_type == ModelType.SCORE:
            drift_fn = score_ode
        elif self.model_type == ModelType.FFD:
            drift_fn = ffd_ode
        else:
            drift_fn = velocity_ode
        
        def body_fn(x, t, model, **model_kwargs):
            model_output = drift_fn(x, t, model, **model_kwargs)
            return model_output

        return body_fn
    

    def get_score(
        self,
    ):
        """member function for obtaining score of 
            x_t = alpha_t * x + sigma_t * eps"""
        if self.model_type == ModelType.NOISE:
            score_fn = lambda x, t, model, **kwargs: model(x, t, **kwargs) / -self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))[0]
        elif self.model_type == ModelType.SCORE:
            score_fn = lambda x, t, model, **kwagrs: model(x, t, **kwagrs)
        elif self.model_type == ModelType.VELOCITY:
            score_fn = lambda x, t, model, **kwargs: self.path_sampler.get_score_from_velocity(model(x, t, **kwargs), x, t)
        elif self.model_type == ModelType.FFD:
            score_fn = lambda x, t, model, **kwargs: model(x, t, **kwargs)
        else:
            raise NotImplementedError()
        
        return score_fn


class Sampler:
    """Sampler class for the transport model"""
    def __init__(
        self,
        transport,
    ):
        """Constructor for a general sampler; supporting different sampling methods
        Args:
        - transport: an tranport object specify model prediction & interpolant type
        """
        
        self.transport = transport
        self.drift = self.transport.get_drift()
        self.score = self.transport.get_score()
    
    def __get_sde_diffusion_and_drift(
        self,
        *,
        diffusion_form="SBDM",
        diffusion_norm=1.0,
    ):

        def diffusion_fn(x, t):
            diffusion = self.transport.path_sampler.compute_diffusion(x, t, form=diffusion_form, norm=diffusion_norm)
            return diffusion
        
        sde_drift = \
            lambda x, t, model, **kwargs: \
                self.drift(x, t, model, **kwargs) + diffusion_fn(x, t) * self.score(x, t, model, **kwargs)
    
        sde_diffusion = diffusion_fn

        return sde_drift, sde_diffusion
    
    def __get_last_step(
        self,
        sde_drift,
        *,
        last_step,
        last_step_size,
    ):
        """Get the last step function of the SDE solver"""
    
        if last_step is None:
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x
        elif last_step == "Mean":
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x + sde_drift(x, t, model, **model_kwargs) * last_step_size
        elif last_step == "Tweedie":
            alpha = self.transport.path_sampler.compute_alpha_t # simple aliasing; the original name was too long
            sigma = self.transport.path_sampler.compute_sigma_t
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x / alpha(t)[0][0] + (sigma(t)[0][0] ** 2) / alpha(t)[0][0] * self.score(x, t, model, **model_kwargs)
        elif last_step == "Euler":
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x + self.drift(x, t, model, **model_kwargs) * last_step_size
        else:
            raise NotImplementedError()

        return last_step_fn

    def sample_sde(
        self,
        *,
        sampling_method="Euler",
        diffusion_form="SBDM",
        diffusion_norm=1.0,
        last_step="Mean",
        last_step_size=0.04,
        num_steps=250,
    ):
        """returns a sampling function with given SDE settings
        Args:
        - sampling_method: type of sampler used in solving the SDE; default to be Euler-Maruyama
        - diffusion_form: function form of diffusion coefficient; default to be matching SBDM
        - diffusion_norm: function magnitude of diffusion coefficient; default to 1
        - last_step: type of the last step; default to identity
        - last_step_size: size of the last step; default to match the stride of 250 steps over [0,1]
        - num_steps: total integration step of SDE
        """

        if last_step is None:
            last_step_size = 0.0

        sde_drift, sde_diffusion = self.__get_sde_diffusion_and_drift(
            diffusion_form=diffusion_form,
            diffusion_norm=diffusion_norm,
        )

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            diffusion_form=diffusion_form,
            sde=True,
            eval=True,
            reverse=False,
            last_step_size=last_step_size,
        )

        _sde = sde(
            sde_drift,
            sde_diffusion,
            t0=t0,
            t1=t1,
            num_steps=num_steps,
            sampler_type=sampling_method
        )

        last_step_fn = self.__get_last_step(sde_drift, last_step=last_step, last_step_size=last_step_size)
            

        def _sample(init, model, **model_kwargs):
            xs = _sde.sample(init, model, **model_kwargs)
            ts = th.ones(init.size(0), device=init.device) * t1
            x = last_step_fn(xs[-1], ts, model, **model_kwargs)
            xs.append(x)

            assert len(xs) == num_steps, "Samples does not match the number of steps"

            return xs

        return _sample
    
    def sample_ode_inversion(
        self,
        *,
        num_steps=250,
    ):
        """returns a sampling function with given ODE settings
        Args:
        - sampling_method: type of sampler used in solving the ODE; default to be Dopri5
        - num_steps: 
            - fixed solver (Euler, Heun): the actual number of integration steps performed
            - adaptive solver (Dopri5): the number of datapoints saved during integration; produced by interpolation
        - atol: absolute error tolerance for the solver
        - rtol: relative error tolerance for the solver
        - reverse: whether solving the ODE in reverse (data to noise); default to False
        """
        # if reverse:
        #     drift = lambda x, t, model, **kwargs: self.drift(x, th.ones_like(t) * (1 - t), model, **kwargs)
        # else:
        #     drift = self.drift
        drift = self.drift

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=False,
            last_step_size=0.0,
        )

        _ode = ode(
            drift=drift,
            t0=t0,
            t1=t1,
            sampler_type="euler",
            num_steps=num_steps,
            atol=1e-6,
            rtol=1e-3,
            timestep_shift=0.0,
        )
        
        return _ode.inversion_fixed_point_sampling
    
    def sample_ode(
        self,
        *,
        sampling_method="dopri5",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
        reverse=False,
        timestep_shift=0.0,
    ):
        """returns a sampling function with given ODE settings
        Args:
        - sampling_method: type of sampler used in solving the ODE; default to be Dopri5
        - num_steps: 
            - fixed solver (Euler, Heun): the actual number of integration steps performed
            - adaptive solver (Dopri5): the number of datapoints saved during integration; produced by interpolation
        - atol: absolute error tolerance for the solver
        - rtol: relative error tolerance for the solver
        - reverse: whether solving the ODE in reverse (data to noise); default to False
        """
        if self.transport.model_type == ModelType.FFD:
            return lambda x, model, **kwargs: {"preds": [model(x, **kwargs)["pred"]]}
        
        if reverse:
            drift = lambda x, t, model, **kwargs: -self.drift(x, th.ones_like(t) * (1 - t), model, **kwargs)
        else:
            drift = self.drift
        # drift = self.drift

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=False,
            last_step_size=0.0,
        )

        _ode = ode(
            drift=drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
            timestep_shift=timestep_shift,
        )
        
        return _ode.sample

    def guided_sampling_flowchef(
        self,
        *,
        num_steps=50,
    ):
        """returns a sampling function with given ODE settings
        Args:
        - num_steps: 
            - fixed solver (Euler, Heun): the actual number of integration steps performed
        """

        _ode = ode(
            drift=self.drift,
            t0=0,
            t1=1,
            sampler_type="Euler", # unused
            num_steps=num_steps,
            atol=1e-6, # unused
            rtol=1e-3, # unused
            timestep_shift=0, # unused
        )
        
        return _ode.guided_sampling_flowchef
    def guided_sampling_flowdps(
        self,
        *,
        num_steps=50,
    ):
        """returns a sampling function with given ODE settings
        Args:
        - num_steps: 
            - fixed solver (Euler, Heun): the actual number of integration steps performed
        """

        _ode = ode(
            drift=self.drift,
            t0=0,
            t1=1,
            sampler_type="Euler", # unused
            num_steps=num_steps,
            atol=1e-6, # unused
            rtol=1e-3, # unused
            timestep_shift=0, # unused
        )
        
        return _ode.guided_sampling_flowdps

    def sample_ode_likelihood(
        self,
        *,
        sampling_method="dopri5",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
    ):
        
        """returns a sampling function for calculating likelihood with given ODE settings
        Args:
        - sampling_method: type of sampler used in solving the ODE; default to be Dopri5
        - num_steps: 
            - fixed solver (Euler, Heun): the actual number of integration steps performed
            - adaptive solver (Dopri5): the number of datapoints saved during integration; produced by interpolation
        - atol: absolute error tolerance for the solver
        - rtol: relative error tolerance for the solver
        """
        def _likelihood_drift(x, t, model, **model_kwargs):
            x, _ = x
            eps = th.randint(2, x.size(), dtype=th.float, device=x.device) * 2 - 1
            t = th.ones_like(t) * (1 - t)
            with th.enable_grad():
                x.requires_grad = True
                grad = th.autograd.grad(th.sum(self.drift(x, t, model, **model_kwargs) * eps), x)[0]
                logp_grad = th.sum(grad * eps, dim=tuple(range(1, len(x.size()))))
                drift = self.drift(x, t, model, **model_kwargs)
            return (-drift, logp_grad)
        
        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=False,
            last_step_size=0.0,
        )

        _ode = ode(
            drift=_likelihood_drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
        )

        def _sample_fn(x, model, **model_kwargs):
            init_logp = th.zeros(x.size(0)).to(x)
            input = (x, init_logp)
            drift, delta_logp = _ode.sample(input, model, **model_kwargs)
            drift, delta_logp = drift[-1], delta_logp[-1]
            prior_logp = self.transport.prior_logp(drift)
            logp = prior_logp - delta_logp
            return logp, drift

        return _sample_fn



    def sample_ode_with_sde_window(
        self,
        *,
        noise_level: float = 0.7,
        sde_window_size: int = 0,
        sde_window_range: Tuple[int, int] = (0, 5),
        sde_type: Optional[str] = "sde",   # keep interface, currently only "sde"/"cps"
        diffusion_form: str = "SBDM",
        diffusion_norm: float = 1.0,
        sampling_method="euler",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
        reverse=False,
        timestep_shift=0.0,
        exclude_last_step_from_window: bool = True,
    ):
        # Use same ODE grid
        if reverse:
            drift = lambda x, t, model, **kwargs: -self.drift(x, th.ones_like(t) * (1 - t), model, **kwargs)
        else:
            drift = self.drift

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=False,
            last_step_size=0.0,
        )

        _ode = ode(
            drift=drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
            timestep_shift=timestep_shift,
        )

        def _sigma_like_x(t_b_fp32: th.Tensor, x_fp32: th.Tensor) -> th.Tensor:
            sigma_t, _ = self.transport.path_sampler.compute_sigma_t(path.expand_t_like_x(t_b_fp32, x_fp32))
            return sigma_t

        @th.no_grad()
        def _sample(init: th.Tensor, model, **model_kwargs):
            x = init
            device = x.device
            B = x.shape[0]
            x_dtype = x.dtype

            # ALWAYS keep time grid fp32
            ts = _ode.t.to(device=device, dtype=th.float32)  # [num_steps]
            T = len(ts) - 1  # number of transitions (i = 0..T-1)

            # ---- choose sde_window (SD3 aligned) ----
            if sde_window_size > 0:
                low, high = int(sde_window_range[0]), int(sde_window_range[1])
                low = max(low, 0)

                max_i = T
                if exclude_last_step_from_window:
                    max_i = max(T - 1, 0)
                high = min(high, max_i)

                if high - low < sde_window_size:
                    raise ValueError(
                        f"sde_window_range={sde_window_range} too small for sde_window_size={sde_window_size} "
                        f"under T={T}, exclude_last_step_from_window={exclude_last_step_from_window}"
                    )
                random.seed(41)
                np.random.seed(41)
                start = random.randint(low, high - sde_window_size)
                end = start + sde_window_size
                sde_window = (start, end)
            else:
                # window_size=0: 纯 ODE（并且返回 window=(0,0) 便于下游判断）
                sde_window = (0, 0)

            # ---- outputs ----
            preds: List[th.Tensor] = [x]
            all_latents: List[th.Tensor] = []
            all_log_probs: List[th.Tensor] = []
            all_timesteps: List[th.Tensor] = []

            # training alignment helpers（只记录 window 内的 step）
            t_pairs: List[th.Tensor] = []
            t_indices: List[th.Tensor] = []
            t_grid = ts.detach().clone()

            # 预计算 sigma_max（模仿 SD3 scheduler.sigmas[1] 的常用 hack）
            # 这里用 ts[1] 的 sigma 作参考，做成 broadcastable tensor
            idx = 1 if len(ts) > 1 else 0
            t_ref = th.full((B,), float(ts[idx].item()), device=device, dtype=th.float32)
            sigma_ref = _sigma_like_x(t_ref, _as_fp32(x)).detach()  # [B,1,1,1...]
            sigma_max_scalar = sigma_ref  # already broadcastable

            # ---- denoising loop (SD3 aligned window logic) ----
            for i in range(T):
                t_cur = ts[i]
                t_next = ts[i + 1]

                # SD3 window logic
                if sde_window_size <= 0:
                    cur_noise_level = 0.0
                else:
                    if i < sde_window[0]:
                        cur_noise_level = 0.0
                    elif i == sde_window[0]:
                        cur_noise_level = float(noise_level)
                        # SD3: all_latents.append(latents) BEFORE first stochastic step
                        all_latents.append(x)
                    elif i > sde_window[0] and i < sde_window[1]:
                        cur_noise_level = float(noise_level)
                    else:
                        cur_noise_level = 0.0

                # batch timesteps (fp32)
                t_b = th.full((B,), float(t_cur.item()), device=device, dtype=th.float32)
                t_next_b = th.full((B,), float(t_next.item()), device=device, dtype=th.float32)


                # # ---- stochastic step (SDE window) ----
                if sde_type != "sde":
                    raise ValueError(f"Only sde_type='sde' is implemented in this aligned version, got {sde_type}")

                x_f = _as_fp32(x)

                # model drift in t-space: dx/dt
                out = drift(x, t_b.to(x_dtype), model, **model_kwargs)
        
                v_t = _as_fp32(_get_pred(out))

                v_sigma = v_t

                # ddpo-like step + logprob (all fp32)
                x_next_f, log_prob, _, _ = ddpo_transport_step_with_logprob(
                    transport=self.transport,
                    model_output=v_sigma,
                    t_cur=t_b,
                    t_next=t_next_b,
                    sample=x_f,
                    noise_level=cur_noise_level,
                    prev_sample=None,
                    sigma_max_fallback=sigma_max_scalar,
                )
                # print('v_sigma shape', v_sigma.shape)
                # print('t_next_b shape', t_next_b.shape)
                # x_next_f = x_f + v_sigma * (t_next_b - t_b).unsqueeze(-1)

                x = x_next_f.to(dtype=x_dtype)
                preds.append(x)

                # SD3: window 内记录（注意 start 时我们已经 append 过“window 前 latents”）
                if i >= sde_window[0] and i < sde_window[1]:
                    all_latents.append(x)
                    all_log_probs.append(log_prob.to(device=device, dtype=th.float32))
                    all_timesteps.append(t_cur.to(device=device, dtype=th.float32))

                    t_pairs.append(th.stack([t_cur, t_next], dim=0))
                    t_indices.append(th.tensor(i, device=device, dtype=th.long))

            return {
                "preds": preds,
                "all_latents": all_latents,
                "all_log_probs": all_log_probs,
                "all_timesteps": all_timesteps,
                "sde_window": sde_window,
                "t_grid": t_grid,  # fp32 [num_steps]
                "t_indices": th.stack(t_indices, dim=0) if len(t_indices) > 0 else None,  # [T_window]
                "t_pairs": th.stack(t_pairs, dim=0) if len(t_pairs) > 0 else None,        # [T_window,2]
            }

        return _sample




@th.no_grad()
def rollout_group_ddpo(
    model,
    transport,
    x_target: th.Tensor,          # [B,...]
    cond_kwargs: Dict[str, Any],
    cfg: DictConfig,
    device: th.device,
    eval: bool = False,
):
    """
    Rollout using Sampler.sample_ode_with_sde_window.

    Returns:
      latents: [B,G,T+1,...]
      logps:   [B,G,T]
      t_steps: [T] sampler-provided timesteps list (usually t_cur for each transition)
      meta:
        - t_pairs: [T,2] true (t_cur, t_next) per transition (preferred)
        - t_grid:  [num_steps] if available
        - start_index, sde_window, num_steps
    """
    model.eval()
    B = x_target.shape[0]

    if eval:
        G = 1
        num_steps = int(getattr(cfg.ddpo, "num_sampling_steps_ddpo_eval", 50))
        sde_window_size = 0
        sde_window_range = (0, num_steps - 1)
        # sde_window_range = (0, num_steps)
        noise_level = 0.0
    else:
        G = int(getattr(cfg.ddpo, "group_size", 16))
        num_steps = int(getattr(cfg.ddpo, "num_sampling_steps_ddpo_train", 50))
        sde_window_size = int(getattr(cfg.ddpo, "sde_window_size", 1))
        sde_window_range = tuple(getattr(cfg.ddpo, "sde_window_range", (0, num_steps // 2))) # sde_window start in [low, high)
        noise_level = float(getattr(cfg.ddpo, "noise_level", 0.7))

    sde_type = str(getattr(cfg.ddpo, "sde_type", "sde"))

    # Expand cond to [B*G,...]
    cond_BG: Dict[str, Any] = {}
    for k, v in cond_kwargs.items():
        if isinstance(v, th.Tensor) and v.shape[0] == B:
            cond_BG[k] = v[:, None, ...].expand(B, G, *v.shape[1:]).reshape(B * G, *v.shape[1:])
        else:
            cond_BG[k] = v

    
    sampler = Sampler(transport)

    sample_fn = sampler.sample_ode_with_sde_window(
        noise_level=noise_level,
        sde_window_size=sde_window_size,
        sde_window_range=sde_window_range,
        sde_type=sde_type,
        sampling_method=cfg.sample.sampling_method,
        num_steps=num_steps,
        atol=1e-6,
        rtol=1e-3,
        reverse=False,
        timestep_shift=float(getattr(cfg.sample, "timestep_shift", 0.0)),
        exclude_last_step_from_window=True,
    )
    sample_fn_ode = sampler.sample_ode(
        sampling_method=cfg.sample.sampling_method,
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
        reverse=False,
        timestep_shift=float(getattr(cfg.sample, "timestep_shift", 0.0)),
    )

    # th.manual_seed(43)
    # th.cuda.manual_seed_all(43)

    # import time
    # seed = time.time_ns() % (2**32)  # torch 

    # th.manual_seed(seed)
    # th.cuda.manual_seed_all(seed)


    if cfg.ddpo.same_noise_within_group:
        # same seed for each data batch
        x_init_base = th.randn((B, *x_target.shape[1:]), device=device, dtype=x_target.dtype)
        x_init = x_init_base.repeat_interleave(G, dim=0)

    else:
        # completely different seed
        x_init = th.randn((B * G, *x_target.shape[1:]), device=device, dtype=x_target.dtype)
    



    autocast_ctx = build_autocast_ctx(cfg)

    if eval:
        with autocast_ctx:
            out = sample_fn_ode(x_init, model, **cond_BG)

        all_preds = out                 # list, len == num_steps (含 init)
        # all_latents = out["all_latents"]
        all_preds = all_preds.transpose(1, 0)
        # all_preds = th.stack(all_preds, dim=1).to(device=device)   # [BG, S, ...]
        # all_latents = th.stack(all_latents, dim=1).to(device=device)   # [BG, T+1, ...]
        S = all_preds.shape[1]                                        # 应该 == num_steps
        all_preds = all_preds.view(B, G, S, *x_target.shape[1:])

        return all_preds, None, None, None
        # return latents.view(B, G, -1, *x_target.shape[1:]), th.stack(out_ode['preds'], dim=1).view(B, G, -1, *x_target.shape[1:]), None, None

    with autocast_ctx:
        out = sample_fn(x_init, model, **cond_BG)

    all_preds = out["preds"]                 # list, len == num_steps (含 init)
    all_latents = out["all_latents"]         # list, len == T_window+1
    all_logps = out["all_log_probs"]         # list, len == T_window
    tp = out.get("t_pairs", None)            # tensor [T_window,2] or list

    # ---- 1) stack preds: [B*G, num_steps, ...]
    all_preds = th.stack(all_preds, dim=1).to(device=device)   # [BG, S, ...]
    S = all_preds.shape[1]                                        # 应该 == num_steps
    all_preds = all_preds.view(B, G, S, *x_target.shape[1:])

    # ---- 2) stack window traj/logp: [B*G, T_window+1,...], [B*G, T_window]
    latents = th.stack(all_latents, dim=1).to(device=device)   # [BG, T+1, ...]
    logps   = th.stack(all_logps, dim=1).to(device=device)     # [BG, T]

    T = logps.shape[1]                                            # T_window
    latents = latents.view(B, G, T + 1, *x_target.shape[1:])
    logps   = logps.view(B, G, T)

    # ---- 3) t_pairs: 必须来自 sampler（不再 fallback）
    if tp is None:
        raise KeyError("Sampler must return out['t_pairs'] after your _sample change.")
    if isinstance(tp, list):
        t_pairs = th.stack([th.stack([a, b]) for (a, b) in tp], dim=0)
    else:
        t_pairs = tp
    t_pairs = t_pairs.to(device=device, dtype=th.float32)      # [T,2]
    assert t_pairs.shape[0] == T, (t_pairs.shape, T)

    # 如果你还想保留 t_steps，用 t_pairs[:,0] 即可
    t_steps = t_pairs[:, 0].contiguous()

    meta = {
        "t_pairs": t_pairs,                 # [T,2]
        "t_grid": out.get("t_grid", None),  # optional
        "t_indices": out.get("t_indices", None),
        "start_index": int(out["sde_window"][0]),
        "sde_window": (int(out["sde_window"][0]), int(out["sde_window"][1])),
        "num_steps": int(num_steps),
    }
    return all_preds, latents, logps, t_steps, meta


def build_autocast_ctx(cfg) -> contextlib.AbstractContextManager:
    """
    Follow flow_grpo:
      - LoRA training: do NOT use autocast (nullcontext)
      - non-LoRA training: use autocast(dtype = fsdp.param_dtype) when dtype is fp16/bf16
    """
    # enable_lora = getattr(cfg, "peft", None) is not None and bool(getattr(cfg.peft, "enable", False))
    # if enable_lora:
    #     return contextlib.nullcontext()

    amp_dtype = getattr(th, cfg.fsdp.param_dtype)
    if amp_dtype in (th.float16, th.bfloat16):
        return th.autocast(device_type="cuda", dtype=amp_dtype)
    return contextlib.nullcontext()


def logprob_transition_current(
    model,
    *,
    transport,
    x_t: th.Tensor,         # [N,...]
    x_next: th.Tensor,      # [N,...]
    t_cur: th.Tensor,       # [N]
    t_next: th.Tensor,      # [N]
    cond_kwargs: Dict[str, Any],
    cfg: DictConfig,
) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    """
    log πθ(x_next | x_t) using transport-based DDPO math.
    Fix: force sigma/logprob path to fp32 (disable autocast).
    """
    amp_dtype = getattr(th, cfg.fsdp.param_dtype)
    model_autocast = build_autocast_ctx(cfg)

    # model forward can be autocast, but ddpo math should be fp32 without autocast
    print("x_t shape: ", x_t.shape)
    with model_autocast:
        out = model(x_t, t=t_cur, **cond_kwargs)
        model_out = unwrap_model_output(out)

    # print('model out', model_out)
    with th.autocast(device_type="cuda", enabled=False):
        _, logp, mean, eff_std = ddpo_transport_step_with_logprob(
            transport,
            model_output=model_out,
            t_cur=t_cur,
            t_next=t_next,
            sample=x_t,
            noise_level=float(getattr(cfg.ddpo, "noise_level", 0.7)),
            prev_sample=x_next,  # crucial: force next
            sigma_max_fallback=None,
            sigma_clip_eps=float(getattr(cfg.ddpo, "sigma_clip_eps", 1e-5)),
        )

    return logp, mean, eff_std



# -----------------------------------------------------------------------------
# Unit tests (fp16/bf16 safe)
# -----------------------------------------------------------------------------

def _assert_finite_tensor(x: th.Tensor, name: str):
    if not th.isfinite(x).all():
        raise AssertionError(f"{name} has NaN/Inf: dtype={x.dtype}, min={th.nan_to_num(x).min().item()}, max={th.nan_to_num(x).max().item()}")

def _assert_finite_list(xs, name: str):
    for i, x in enumerate(xs):
        _assert_finite_tensor(x, f"{name}[{i}]")

def _first_mismatch(preds_a, preds_b, atol=0.0, rtol=0.0):
    for i in range(len(preds_a)):
        a, b = preds_a[i], preds_b[i]
        diff = (a - b).abs()
        tol = atol + rtol * b.abs()
        if (diff > tol).any():
            return i
    return None



def main():
    th.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    # dtype = th.float16  # <-- 你要的 float16；也可改 bfloat16
    dtype = th.float32  # <-- 你要的 float16；也可改 bfloat16

    transport = Transport(
        model_type="velocity",
        path_type="Linear",
        loss_type="none",
        train_eps=0.0,
        sample_eps=0.0,
    )
    sampler = Sampler(transport)

    # toy model: returns dict {"pred": ...}
    a, b = 0.1, 0.3
    def model(x, t=None, **kwargs):
        if t is None:
            t_term = 0.0
        else:
            # t is fp32 [B]; broadcast to x
            t_term = t.view(-1, *([1] * (x.ndim - 1))).to(dtype=th.float32, device=x.device)
        # compute in fp32 then cast back is common in real AMP models; here keep simple
        v = a * _as_fp32(x) + b * t_term
        return {"pred": v.to(dtype=x.dtype)}

    B, D = 4, 128
    x0 = th.randn(B, D, device=device, dtype=dtype)
    num_steps = 31

    # baseline ode(euler)
    ode_fn = sampler.sample_ode(sampling_method="euler", num_steps=num_steps)
    out_ode = ode_fn(x0.clone(), model)
    preds_ode = out_ode["preds"]
    _assert_finite_list(preds_ode, "preds_ode")

    # Test 1: window disabled => exactly equal
    mix0_fn = sampler.sample_ode_with_sde_window(
        sampling_method="euler",
        num_steps=num_steps,
        sde_window_size=0,
        sde_window_range=(0, num_steps),
        noise_level=0.7,
    )
    out_mix0 = mix0_fn(x0.clone(), model)
    preds_mix0 = out_mix0["preds"]
    mm = _first_mismatch(preds_mix0, preds_ode, atol=0.0, rtol=0.0) # failed due to odeint vs eular integral
    # mm = _first_mismatch(preds_mix0, preds_ode, atol=1e-5, rtol=1e-5)
    if mm is not None:
        raise AssertionError(f"FAIL Test1: mismatch at step {mm} when window disabled")

    # Test 2: window enabled but noise_level=0 => must still equal
    mix_deg_fn = sampler.sample_ode_with_sde_window(
        sampling_method="euler",
        num_steps=num_steps,
        sde_window_size=5,
        sde_window_range=(3, num_steps - 3),
        noise_level=0.0,
    )
    out_deg = mix_deg_fn(x0.clone(), model)
    preds_deg = out_deg["preds"]
    # mm = _first_mismatch(preds_deg, preds_ode, atol=1e-5, rtol=1e-5)
    mm = _first_mismatch(preds_deg, preds_ode, atol=0.0, rtol=0.0)
    if mm is not None:
        raise AssertionError(f"FAIL Test2: mismatch at step {mm} when noise_level=0 (should degenerate)")

    # Test 3: window enabled and noise>0 => should diverge after window start
    # th.manual_seed(123); random.seed(123)
    mix_noise_fn = sampler.sample_ode_with_sde_window(
        sampling_method="euler",
        num_steps=num_steps,
        sde_window_size=5,
        sde_window_range=(3, num_steps - 3),
        noise_level=0.7,
        sde_type="sde",
    )
    out_noise = mix_noise_fn(x0.clone(), model)
    preds_noise = out_noise["preds"]
    _assert_finite_list(preds_noise, "preds_noise")
    for lp in out_noise["all_log_probs"]:
        _assert_finite_tensor(lp, "log_prob")

    print("ALL TESTS PASSED (float16 safe).")

# unit test of sample_ode_with_sde_window: python -m src.transport.transport
if __name__ == "__main__":
    main()