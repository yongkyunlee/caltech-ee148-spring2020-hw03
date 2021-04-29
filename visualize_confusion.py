
import os

import torch
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from main import Net
from data_augmentation import augmentation_scheme

from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay

MODEL_PATH = './model'

def get_confusion_matrix(model, device, test_loader):
    # Initialize the prediction and label lists(tensors)
    pred_arr = torch.zeros(0, dtype=torch.long, device='cpu')
    target_arr =torch.zeros(0, dtype=torch.long, device='cpu')
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
            # Append batch prediction results
            pred_arr = torch.cat([pred_arr, pred.view(-1).cpu()])
            target_arr = torch.cat([target_arr, target.view(-1).cpu()])

    test_loss /= test_num

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, test_num,
        100. * correct / test_num))

    return confusion_matrix(target_arr.numpy(), pred_arr.numpy())

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
        test_dataset, batch_size=32, **kwargs)

    confusion_matrix = get_confusion_matrix(model, device, test_loader)

    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                  display_labels=np.array([i for i in range(10)]))
    plt.figure()
    disp.plot()
    plt.savefig('confusion_matrix.png')
