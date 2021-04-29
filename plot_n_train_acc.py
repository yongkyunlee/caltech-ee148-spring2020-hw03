import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    train_size_arr = np.array([51000, 25500, 12750, 6375, 3187])
    train_loss_arr = np.array([0.0446, 0.0558, 0.0803, 0.1260, 0.1286])
    train_acc_arr = np.array([8893/9000, 4435/4500, 2200/2250, 1081/1125, 545/562])
    test_loss_arr = np.array([0.0386, 0.0486, 0.0701, 0.0916, 0.1335])
    test_acc_arr = np.array([9875/10000, 9869/10000, 9800/10000, 9749/10000, 9608/10000])

    fig = plt.figure()
    plt.plot(np.log(train_size_arr), np.log(train_loss_arr))
    plt.plot(np.log(train_size_arr), np.log(test_loss_arr))
    plt.title('Loss for training set size in log-log scale')
    plt.xlabel('log(Training Set Size)')
    plt.ylabel('log(Loss)')
    plt.legend(['Training Loss', 'Test Loss'])

    fig.savefig('./loss_plot.png')

    fig = plt.figure()
    plt.plot(np.log(train_size_arr), np.log(train_acc_arr))
    plt.plot(np.log(train_size_arr), np.log(test_acc_arr))
    plt.title('Accuracy as a function of training set size in log-log scale')
    plt.xlabel('log(Training Set Size)')
    plt.ylabel('log(Accuracy)')
    plt.legend(['Training Accuracy', 'Test Accuracy'])
    fig.savefig('./acc_plot.png')