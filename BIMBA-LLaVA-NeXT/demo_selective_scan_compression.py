import torch
from torch import nn
from llava.model.multimodal_resampler.mamba_ssm.modules.mamba_compressor import MambaCompressor

model = MambaCompressor(d_model=1024, n_layer=1).to("cuda")
torch.nn.init.constant_(model.layers[-1].mixer.out_proj.weight, 0)

for n, p in model.named_parameters():
    if hasattr(p, "ds_numel"):
        print(n, torch.sum(p.ds_tensor).item())
    else:
        print(n, torch.sum(p).item())

batch_size = 1
input_shape = (64, 24, 24)
hidden_size = 1024
target_shape = (16, 12, 12)
temporal_pooling = False
pooling = nn.AdaptiveAvgPool3d(target_shape)

space_time_tokens = torch.randn(batch_size, input_shape[0], input_shape[1], input_shape[2], hidden_size).to("cuda")
if not temporal_pooling:
    query_tokens = space_time_tokens[:,::4]
# [1, 16, 24, 24, 1024]
query_tokens = query_tokens.permute(0, 4, 1, 2, 3)
# [1, 1024, 16, 24, 24]
query_tokens = pooling(query_tokens)
# [1, 1024, 16, 12, 12]
query_tokens = query_tokens.permute(0, 2, 3, 4, 1)
# [1, 16, 12, 12, 1024]
query_tokens = query_tokens.reshape(batch_size, target_shape[0], -1, hidden_size)
print(space_time_tokens.shape, query_tokens.shape)


query_tokens = model(space_time_tokens, query_tokens)
query_tokens = query_tokens.reshape(batch_size,target_shape[0], target_shape[1], target_shape[2], hidden_size)
print(query_tokens.shape)