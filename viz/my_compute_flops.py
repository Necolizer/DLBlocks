import torch
from calflops import calculate_flops
from model.ISTANet import ModelAlign
from model.Align import Align

model = ModelAlign(
    align_mlp=Align(768, 256),
    window_size= [20, 1, 2],
    num_frames=120,
    num_joints=25,
    num_persons=2,
    num_channels=3,
    num_classes=26,
    num_heads=3,
    kernel_size=[3, 5],
    use_pes=True,
    config=[[64,  64,  16], [64,  64,  16], 
            [64,  128, 32], [128, 128, 32],
            [128, 256, 64], [256, 256, 64], 
            [256, 256, 64], [256, 256, 64]],
    align_layer=5,
    align_num_channels=256,
)

inputs ={
    'x': torch.rand((1, 3, 120, 25, 2)),
    'vision_feature': torch.rand((1, 768)),
}

flops, macs, params = calculate_flops(model=model, kwargs=inputs, print_results=False)
print(flops)
print(macs)
print(params)