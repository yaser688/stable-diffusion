from torch import nn
from k_diffusion import utils as k_utils
import torch
from k_diffusion.external import CompVisDenoiser
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

class CFGDenoiserWithGrad(CompVisDenoiser):
    def __init__(self, model, loss_fns_scales, 
                       clamp_func=None, gradient_wrt=None, gradient_add_to=None, cond_uncond_sync=True, 
                       decode_method=None,
                       verbose=False):
        super().__init__(model.inner_model)
        self.inner_model = model
        self.clamp_func = clamp_func # Gradient clamping function, clamp_func(grad, sigma)
        self.gradient_wrt = gradient_wrt # Calculate gradient with respect to ["x", "x0_pred", "both"]
        self.gradient_add_to = gradient_add_to # Add gradient to ["cond", "uncond", "both"]
        self.cond_uncond_sync = cond_uncond_sync # Calculates the cond and uncond simultaneously

        # decode_fn is the function used to decode the latent during gradient calculation
        if args.decode_method is None:
            decode_fn = lambda x: x
        elif args.decode_method == "autoencoder":
            decode_fn = model.inner_model.differentiable_decode_first_stage
        elif args.decode_method == "linear":
            decode_fn = model.inner_model.simple_decode

        self.decode_fn = decode_fn
        self.verbose = verbose

        cond_fns = []
        for loss_fn,scale in loss_fns_scales:
            if scale != 0:
                cond_fn = self.make_cond_fn(loss_fn, scale, wrt=gradient_wrt, decode_fn=decode_fn, verbose=False)
            else:
                cond_fn = None
            cond_fns += [cond_fn]
        self.cond_fns = cond_fns

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

        # Clamp the gradient
        total_cond_grad = self.clamp_grad_verbose(total_cond_grad, sigma)

        if self.gradient_wrt == 'x':
            x.copy_(x.detach() + total_cond_grad * k_utils.append_dims(sigma, x.ndim))
            cond_denoised = denoised.detach()
        elif self.gradient_wrt == 'x0_pred':
            cond_denoised = denoised.detach() + total_cond_grad * k_utils.append_dims(sigma, x.ndim)

        # cond_denoised = denoised.detach() + total_cond_grad * k_utils.append_dims(sigma ** 2, x.ndim)

        return cond_denoised

    def cfg_cond_model_fn_(self, x_in, sigma, cond, cond_scale, **kwargs):
        
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])

        total_cond_grad = torch.zeros_like(x)
        for cond_fn in self.cond_fns:
            if cond_fn is None: continue
            if self.gradient_wrt == 'x':
                with torch.enable_grad():
                    x = x_in.detach().requires_grad_()
                    x = torch.cat([x] * 2)
                    denoised = self.inner_model(x, sigma, cond=cond, **kwargs)
                    uncond_x0, cond_x0 = denoised.chunk(2)
                    x0_pred = uncond_x0 + (cond_x0 - uncond_x0) * cond_scale
                    cond_grad = cond_fn(x, sigma, denoised=x0_pred, **kwargs).detach()
            elif self.gradient_wrt == 'x0_pred':
                with torch.no_grad():
                    x = torch.cat([x] * 2)
                    denoised = self.inner_model(x, sigma, **kwargs)
                    uncond_x0, cond_x0 = denoised.chunk(2)
                    x0_pred = uncond_x0 + (cond_x0 - uncond_x0) * cond_scale
                with torch.enable_grad():
                    cond_grad = cond_fn(x, sigma, denoised=x0_pred.detach().requires_grad_(), **kwargs).detach()
            total_cond_grad += cond_grad
        
        total_cond_grad = torch.nan_to_num(total_cond_grad, nan=0.0, posinf=float('inf'), neginf=-float('inf'))

        total_cond_grad = self.clamp_grad_verbose(total_cond_grad, sigma)

        if self.gradient_wrt == 'x':
            x.copy_(x.detach() + total_cond_grad * k_utils.append_dims(sigma, x.ndim))
            x0 = x0_pred.detach()
        elif self.gradient_wrt == 'x0_pred':
            x0 = x0_pred.detach() + total_cond_grad * k_utils.append_dims(sigma, x.ndim)

        return x0

    def forward(self, x, sigma, uncond, cond, cond_scale):
        # Conditioning
        if (self.cond_fns is not None and 
            any(cond_fn is not None for cond_fn in self.cond_fns)):

            # Apply the conditioning gradient to both cond and uncond
            if self.cond_uncond_sync or self.gradient_add_to == "both":
                x0 = self.cfg_cond_model_fn(x, sigma, cond=cond, cond_scale=cond_scale)

            # Calculate cond and uncond separately
            else:
                if self.gradient_add_to == "uncond":
                    uncond = self.cond_model_fn_(x, sigma, cond=uncond)
                    cond = self.inner_model(x, sigma, cond=cond)
                    x0 = uncond + (cond - uncond) * cond_scale
                elif self.gradient_add_to == "cond":
                    uncond = self.inner_model(x, sigma, cond=uncond)
                    cond = self.cond_model_fn_(x, sigma, cond=cond)
                    x0 = uncond + (cond - uncond) * cond_scale
                else: 
                    raise Exception(f"Unrecognised option for gradient_add_to: {self.gradient_add_to}")

        # No conditioning
        else:
            if self.cond_uncond_sync:
                x_in = torch.cat([x] * 2)
                sigma_in = torch.cat([sigma] * 2)
                cond_in = torch.cat([uncond, cond])
                uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
            else:
                uncond = self.inner_model(x, sigma, cond=uncond)
                cond = self.inner_model(x, sigma, cond=cond)

            x0 = uncond + (cond - uncond) * cond_scale

        return x0

    def make_cond_fn(self, loss_fn, scale, wrt='x', decode_fn=None, verbose=False):
        # Turns a loss function into a cond function that is applied to the decoded RGB sample
        # loss_fn (function): func(x, sigma, denoised) -> number
        # scale (number): how much this loss is applied to the image
        # wrt (str): ['x','x0_pred'] get the gradient with respect to this variable, default x
        # decode_fn (callable): method to decode the latent to an image
        if wrt is None:
            wrt = 'x'

        if decode_fn is None:
            decode_fn = lambda x: x

        def cond_fn(x, sigma, denoised, **kwargs):
            with torch.enable_grad():
                denoised_sample = decode_fn(denoised).requires_grad_()
                loss = loss_fn(denoised_sample, sigma, **kwargs) * scale
                grad = -torch.autograd.grad(loss, x)[0]
            verbose_print('Loss:', loss.item())
            return grad

        def cond_fn_pred(x, sigma, denoised, **kwargs):
            with torch.enable_grad():
                denoised_sample = decode_fn(denoised).requires_grad_()
                loss = loss_fn(denoised_sample, sigma, **kwargs) * scale
                grad = -torch.autograd.grad(loss, denoised)[0]
            verbose_print('Loss:', loss.item())
            return grad

        verbose_print = print if verbose else lambda *args, **kwargs: None
        
        if wrt == 'x':
            return cond_fn
        elif wrt == 'x0_pred':
            return cond_fn_pred
        else:
            raise Exception(f"Variable wrt == {wrt} not recognised.")

    def clamp_grad_verbose(self, grad, sigma):
        if self.clamp_func is not None:
            if self.verbose:
                print("Grad before clamping:")
                self.display_images(torch.abs(grad*2.0) - 1.0)
            grad = self.clamp_func(grad, sigma)
        if self.verbose:
            print("Conditioning gradient")
            self.display_images(torch.abs(grad*2.0) - 1.0)
        return grad

    def display_images(self, images):
        images = images.double().cpu().add(1).div(2).clamp(0, 1)
        images = torch.tensor(images.numpy())
        grid = make_grid(images, 4).cpu()
        display.display(to_pil_image(grid))
        return