import os
import torch
from .simulacra_fit_linear_model import AestheticMeanPredictionLinearModel

def load_aesthetics_model(args,root):

    clip_size = {
        "ViT-B/32": 512,
        "ViT-B/16": 512,
        "ViT-L/14": 768,
        "ViT-L/14@336px": 768,
    }

    model_name = {
        "ViT-B/32": "sac_public_2022_06_29_vit_b_32_linear.pth",
        "ViT-B/16": "sac_public_2022_06_29_vit_b_16_linear.pth",
        "ViT-L/14": "sac_public_2022_06_29_vit_l_14_linear.pth",
    }
    
    aesthetics_model = AestheticMeanPredictionLinearModel(clip_size[args.clip_name])
    aesthetics_model.load_state_dict(torch.load(os.path.join(root.models_path,model_name[args.clip_name])))

    return aesthetics_model.to(root.device)