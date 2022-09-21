from typing import Any, Callable, Optional
import torch
from k_diffusion.external import CompVisDenoiser
from k_diffusion import sampling
from k_diffusion import utils as k_utils
from torch import nn
# Display functions
from torchvision.utils import make_grid
from IPython import display
from torchvision.transforms.functional import to_pil_image

class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cond_scale

class CFGDenoiserWithGrad(nn.Module):
    def __init__(self, model, cond_fns, clamp_func=None, gradient_wrt=None, gradient_add_to=None, cond_uncond_sync=True, verbose=False):
        super().__init__()
        self.inner_model = model
        self.cond_fns = cond_fns
        self.clamp_func = clamp_func # Gradient clamping function, clamp_func(grad, sigma)
        self.gradient_wrt = gradient_wrt # Calculate gradient with respect to ["x", "x0_pred", "both"]
        self.gradient_add_to = gradient_add_to # Add gradient to ["cond", "uncond", "both"]
        self.cond_uncond_sync = cond_uncond_sync # Calculates the cond and uncond simultaneously
        self.verbose = verbose

    def cond_model_fn_(self, x, sigma, **kwargs):
        total_cond_grad = torch.zeros_like(x)
        for cond_fn in self.cond_fns:
            if cond_fn is None: continue
            if self.gradient_wrt == 'x':
                with torch.enable_grad():
                    x = x.detach().requires_grad_()
                    denoised = self.inner_model(x, sigma, **kwargs)
                    cond_grad = cond_fn(x, sigma, denoised=denoised, **kwargs).detach()
            elif self.gradient_wrt == 'x0_pred':
                with torch.no_grad():
                    denoised = self.inner_model(x, sigma, **kwargs)
                with torch.enable_grad():
                    cond_grad = cond_fn(x, sigma, denoised=denoised.detach().requires_grad_(), **kwargs).detach()
            total_cond_grad += cond_grad
        
        total_cond_grad = torch.nan_to_num(total_cond_grad, nan=0.0, posinf=float('inf'), neginf=-float('inf'))

        if self.clamp_func is not None:
            if self.verbose:
                print("Grad before clamping:")
                self.display_images(torch.abs(total_cond_grad*2.0) - 1.0)
            total_cond_grad = self.clamp_func(total_cond_grad, sigma)
        if self.verbose:
            print("Conditioning gradient")
            self.display_images(torch.abs(total_cond_grad*2.0) - 1.0)

        if self.gradient_wrt == 'x':
            x.copy_(x.detach() + total_cond_grad * k_utils.append_dims(sigma, x.ndim))
            cond_denoised = denoised.detach()
        elif self.gradient_wrt == 'x0_pred':
            cond_denoised = denoised.detach() + total_cond_grad * k_utils.append_dims(sigma, x.ndim)

        # cond_denoised = denoised.detach() + total_cond_grad * k_utils.append_dims(sigma ** 2, x.ndim)

        return cond_denoised

    def display_images(self, images):
        images = images.double().cpu().add(1).div(2).clamp(0, 1)
        images = torch.tensor(images.numpy())
        grid = make_grid(images, 4).cpu()
        display.display(to_pil_image(grid))
        return

    def forward(self, x, sigma, uncond, cond, cond_scale):        
        if (self.cond_fns is not None and 
            any(cond_fun is not None for cond_fun in self.cond_fns)):
            # Conditioning functions are set, so use conditioning functions
            if self.cond_uncond_sync:
                print(f"Variable cond_uncond_sync == {self.cond_uncond_sync}, so we are applying the conditioning gradient to both cond and uncond.")
                x_in = torch.cat([x] * 2)
                sigma_in = torch.cat([sigma] * 2)
                cond_in = torch.cat([uncond, cond])
                uncond, cond = self.cond_model_fn(x_in, sigma_in, cond=cond_in).chunk(2)
            else: # calculate cond and uncond separately
                if self.gradient_add_to == "both" or self.gradient_add_to == "uncond":
                    uncond = self.cond_model_fn_(x, sigma, cond=uncond)
                else:
                    uncond = self.inner_model(x, sigma, cond=uncond)
                if self.gradient_add_to == "both" or self.gradient_add_to == "cond":
                    cond = self.cond_model_fn_(x, sigma, cond=cond)
                else:
                    cond = self.inner_model(x, sigma, cond=cond)
        else: # No conditioning
            if self.cond_uncond_sync:
                x_in = torch.cat([x] * 2)
                sigma_in = torch.cat([sigma] * 2)
                cond_in = torch.cat([uncond, cond])
                uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
            else:
                uncond = self.inner_model(x, sigma, cond=uncond)
                cond = self.inner_model(x, sigma, cond=cond)

        return uncond + (cond - uncond) * cond_scale

        

def sampler_fn(
    c: torch.Tensor,
    uc: torch.Tensor,
    args,
    model_wrap: CompVisDenoiser,
    init_latent: Optional[torch.Tensor] = None,
    t_enc: Optional[torch.Tensor] = None,
    cond_fns: Optional[list] = None,
    clamp_func: Callable[[Any], None] = None,
    gradient_wrt: Optional[str] = None,
    gradient_add_to: Optional[str] = None,
    cond_uncond_sync: Optional[bool] = True,
    device=torch.device("cpu")
    if not torch.cuda.is_available()
    else torch.device("cuda"),
    cb: Callable[[Any], None] = None,
    verbose: Optional[bool] = False,
) -> torch.Tensor:
    shape = [args.C, args.H // args.f, args.W // args.f]
    sigmas: torch.Tensor = model_wrap.get_sigmas(args.steps)
    sigmas = sigmas[len(sigmas) - t_enc - 1 :]
    if args.use_init:
        if len(sigmas) > 0:
            x = (
                init_latent
                + torch.randn([args.n_samples, *shape], device=device) * sigmas[0]
            )
        else:
            x = init_latent
    else:
        if len(sigmas) > 0:
            x = torch.randn([args.n_samples, *shape], device=device) * sigmas[0]
        else:
            x = torch.zeros([args.n_samples, *shape], device=device)
    sampler_args = {
        "model": CFGDenoiserWithGrad(model_wrap, 
                                    cond_fns, 
                                    clamp_func, 
                                    gradient_wrt, 
                                    gradient_add_to, 
                                    cond_uncond_sync,
                                    verbose=verbose),
        "x": x,
        "sigmas": sigmas,
        "extra_args": {"cond": c, "uncond": uc, "cond_scale": args.scale},
        "disable": False,
        "callback": cb,
    }
    sampler_map = {
        "klms": sampling.sample_lms,
        "dpm2": sampling.sample_dpm_2,
        "dpm2_ancestral": sampling.sample_dpm_2_ancestral,
        "heun": sampling.sample_heun,
        "euler": sampling.sample_euler,
        "euler_ancestral": sampling.sample_euler_ancestral,
    }

    samples = sampler_map[args.sampler](**sampler_args)
    return samples
