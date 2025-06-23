import torch

load_medusa_head = torch.load("/home/nxclab/wonjun/Medusa/medusa_lm_head.pt", map_location="cpu")
print(load_medusa_head['0.0.linear.weight'].shape)