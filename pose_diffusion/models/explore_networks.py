import sys
sys.path.append('./pose_diffusion')

from gaussian_diffuser import GaussianDiffusion
from denoiser import Denoiser, TransformerEncoderWrapper
# from image_feature_extractor import ImageFeatureExtractor

import torch
from easydict import EasyDict as edict

diffuser = GaussianDiffusion()
# transformer_encoder = TransformerEncoderWrapper(d_model=512, nhead=4, num_encoder_layers=8)
TRANSFORMER = {
    "_target_":               TransformerEncoderWrapper(d_model=512, nhead=4, 
                                                        num_encoder_layers=8, dim_feedforward=1024,),
    "d_model":                512,
    "nhead":                  4,
    "dim_feedforward":        1024,
    "num_encoder_layers":     8,
    "dropout":                0.1,
    "batch_first":            True,
    "norm_first":             True,
    
}
TRANSFORMER = edict(TRANSFORMER)
denoiser = Denoiser(TRANSFORMER, pivot_cam_onehot=False)
diffuser.model = denoiser

batch_size, frame_num, target_dim, condition_dim = 4, 1, 9, 384

inputs = torch.rand(batch_size, frame_num ,target_dim)
conditions = torch.rand(batch_size, frame_num ,condition_dim)
diffusion_results = diffuser(inputs, z=conditions)
