import os

from PIL import Image
import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np

from main import Net
from data_augmentation import augmentation_scheme

MODEL_PATH = './model'

def test(model, device, test_loader):
    mistake_idx_arr, mistake_label_arr = [], []
    model.eval()    # Set the model to inference mode
    test_loss = 0
    correct = 0
    test_num = 0
    with torch.no_grad():   # For the inference step, gradient is not computed
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            is_correct = pred.eq(target.view_as(pred))[0][0]
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_num += len(data)
            if not is_correct.item():
                mistake_idx_arr.append(idx)
                mistake_label_arr.append(pred[0][0].item())

    test_loss /= test_num

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, test_num,
        100. * correct / test_num))

    return mistake_idx_arr, mistake_label_arr

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
        test_dataset, batch_size=1, **kwargs)

    mistake_idx_arr, mistake_label_arr = test(model, device, test_loader)
    
    test_dataset = datasets.MNIST('./mnist_data', train=False,
                transform=augmentation_scheme['augment0'])
    for i, idx in enumerate(mistake_idx_arr[:15]):
        img, label = test_dataset[idx]
        img = np.squeeze(img.numpy()) * 255
        # plt.gray()
        # plt.imshow(img)
        img = Image.fromarray(np.uint8(img))
        img.save('./mistakes/mistake_{}_{}.png'.format(idx, mistake_label_arr[i]))
