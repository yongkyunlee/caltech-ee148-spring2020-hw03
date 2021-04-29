import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import utils

from main import Net

MODEL_PATH = './model'

def visTensor(tensor, ch=0, allkernels=False, nrow=4, padding=1): 
    n,c,w,h = tensor.shape

    if c != 3:
        tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

    grid = utils.make_grid(tensor, nrow=nrow, normalize=True,
                           padding=padding)
    plt.figure( figsize=(nrow,nrow) )
    plt.imshow(grid.numpy().transpose((1, 2, 0)))


if __name__ == "__main__":
    model_name = 'model_kernel_4.pt'
    model_path = os.path.join(MODEL_PATH, model_name)
    assert os.path.exists(model_path)

    # Set the test model
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    model = Net().to(device)
    model.load_state_dict(torch.load(model_path))

    kernels = model.conv1.weight.data.clone()
    visTensor(kernels, ch=0, allkernels=False)

    plt.axis('off')
    plt.ioff()
    plt.savefig('./kernels_5.png')