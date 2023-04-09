import numpy as np
import sklearn
import sklearn.model_selection
import torch
from scipy.io import loadmat,savemat


def get_dataset(dataset_name, model_name="ss1dswin", ratio=1):
    if dataset_name == 'Indian':
        data = loadmat('./data/IndianPine.mat')

    elif dataset_name == 'Pavia':
        data = loadmat('./data/Pavia.mat')
        if model_name == "ss1dswin":
            data["input"] = applyInterpolate(data["input"], numComponents=104)
    elif dataset_name == 'Houston':
        data = loadmat('./data/Houston.mat')
    elif dataset_name == 'Salinas':
        data = loadmat('./data/Salinas_corrected.mat')['salinas_corrected']  # 512 217 204
        if model_name == "ss1dswin":
            data = applyInterpolate(data, numComponents=200)
        ALL = loadmat('./data/Salinas_gt.mat')['salinas_gt']  # 512 217
    else:
        raise ValueError("Unkknow dataset")
    if dataset_name != "Salinas":
        input = data['input']
        TE = data['TE']
        TR = data['TR']
    else:
        input = data
        TR, TE = sample_gt(ALL, 67)
    return TE, TR, input


def applyInterpolate(X, numComponents):
    X = X / 1.0
    X = torch.FloatTensor(X)
    X = torch.nn.functional.interpolate(input=X, size=numComponents, mode="linear",
                                        align_corners=True)  # H W numComponents
    X = X.numpy()
    return X


def sample_gt(gt, train_size):
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)

    print("Sampling with train size = {}".format(train_size))
    train_indices, test_indices = [], []
    for c in np.unique(gt):
        if c == 0:
            continue
        indices = np.nonzero(gt == c)
        X = list(zip(*indices))  # x,y features
        train, test = sklearn.model_selection.train_test_split(X, train_size=train_size)
        train_indices += train
        test_indices += test
    train_indices = [list(t) for t in zip(*train_indices)]
    test_indices = [list(t) for t in zip(*test_indices)]
    train_gt[train_indices] = gt[train_indices]
    test_gt[test_indices] = gt[test_indices]
    return train_gt, test_gt
