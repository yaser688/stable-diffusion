from typing import Any, Callable, Optional
from k_diffusion.external import CompVisDenoiser
import torch
from k_diffusion import sampling

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
        "model": model_wrap,
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
