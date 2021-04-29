import numpy as np
import heapq
from sklearn.manifold import TSNE
import os
from PIL import Image

import torch
from torchvision import datasets, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

from main import Net
from data_augmentation import augmentation_scheme

MODEL_PATH = './model'

def get_feature_vectors(model, device, test_loader):
    feature_vector_arr = np.array([]).reshape(0, 64)
    label_arr = []
    # Initialize the prediction and label lists(tensors)
    pred_arr = torch.zeros(0, dtype=torch.long, device='cpu')
    target_arr = torch.zeros(0, dtype=torch.long, device='cpu')
    model.eval()    # Set the model to inference mode
    test_loss = 0
    correct = 0
    test_num = 0
    with torch.no_grad():   # For the inference step, gradient is not computed
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            feature_vector = model.feature_vector(data).numpy()
            feature_vector_arr = np.concatenate(
                (feature_vector_arr, feature_vector), axis=0)

    return feature_vector_arr

def choose_closest_vector(feature_vector_arr, feature_vector):
    h = []
    for idx in range(len(feature_vector_arr)):
        heapq.heappush(h, (-np.linalg.norm(feature_vector_arr[idx] - feature_vector),
                          idx))
        if len(h) > 9: # keep nine because the base image is going to have lowest priority
            heapq.heappop(h)
    return h

if __name__ == '__main__':
    model_name = 'model.pt'
    model_path = os.path.join(MODEL_PATH, model_name)
    assert os.path.exists(model_path)

    torch.manual_seed(1)

    # Set the test model
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    model = Net().to(device)
    model.load_state_dict(torch.load(model_path))

    test_dataset = datasets.MNIST('./mnist_data', train=False,
                transform=augmentation_scheme['augment1'])

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, **kwargs)

    img_idx = 7
    os.makedirs(f'./close_feature_vectors/{img_idx}', exist_ok=True)

    feature_vector_arr = get_feature_vectors(model, device, test_loader)
    feature_vector = feature_vector_arr[img_idx]
    closest_h = choose_closest_vector(feature_vector_arr, feature_vector)
    
    test_dataset = datasets.MNIST('./mnist_data', train=False,
                transform=augmentation_scheme['augment0'])
    for i in range(8):
        _, idx = heapq.heappop(closest_h)
        img, label = test_dataset[idx]
        img = np.squeeze(img.numpy()) * 255
        img = Image.fromarray(np.uint8(img))
        img.save(f'./close_feature_vectors/{img_idx}/img_{i}.png')