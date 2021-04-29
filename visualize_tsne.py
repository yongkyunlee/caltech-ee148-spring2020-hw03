import numpy as np
from sklearn.manifold import TSNE

import os

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

    feature_vectors = get_feature_vectors(model, device, test_loader)
    label_arr = test_dataset.targets.numpy()
    tsne = TSNE(n_components=2, verbose=1, n_iter=1000)
    tsne_results = tsne.fit_transform(feature_vectors, label_arr)

    plt.figure()
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'grey', 'orange', 'purple']
    for idx in range(10):
        plt.scatter(tsne_results[label_arr == idx, 0], tsne_results[label_arr == idx, 1],
                    c=colors[idx], label=idx)
    plt.legend()
    plt.savefig('tsne.png')
