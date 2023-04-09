import argparse
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as Data
from sklearn.metrics import confusion_matrix

from config import get_config
from dataset import get_dataset
from method import SwinT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser("HSI")
parser.add_argument('--dataset', choices=['Indian', 'Pavia', 'Houston', 'Salinas'], default='Indian',
                    help='dataset to use')
parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=0, help='number of seed')

args = parser.parse_args()


def choose_true_point(true_data, num_classes):
    number_true = []
    pos_true = {}
    for i in range(num_classes + 1):
        each_class = np.argwhere(true_data == i)
        number_true.append(each_class.shape[0])
        pos_true[i] = each_class

    total_pos_true = pos_true[0]
    for i in range(1, num_classes + 1):
        total_pos_true = np.r_[total_pos_true, pos_true[i]]
    total_pos_true = total_pos_true.astype(int)
    return total_pos_true, number_true


def choose_img_point(height, width):
    total_pos_true = np.array([[i, j] for i in range(height) for j in range(width)])
    return total_pos_true


# 1
def chooose_point(test_data, num_classes):
    number_test = []
    pos_test = {}

    for i in range(num_classes):
        each_class = np.argwhere(test_data == (i + 1))
        number_test.append(each_class.shape[0])
        pos_test[i] = each_class

    total_pos_test = pos_test[0]
    for i in range(1, num_classes):
        total_pos_test = np.r_[total_pos_test, pos_test[i]]  # (9671,2)
    total_pos_test = total_pos_test.astype(int)
    return total_pos_test, number_test


def mirror_hsi(height, width, band, input_normalize, patch=5):
    padding = patch // 2
    mirror_hsi = np.zeros((height + 2 * padding, width + 2 * padding, band), dtype=float)  # padding后的图 上下左右各加padding

    mirror_hsi[padding:(padding + height), padding:(padding + width), :] = input_normalize  # 中间用原图初始化

    for i in range(padding):
        mirror_hsi[padding:(height + padding), i, :] = input_normalize[:, padding - i - 1, :]

    for i in range(padding):
        mirror_hsi[padding:(height + padding), width + padding + i, :] = input_normalize[:, width - 1 - i, :]

    for i in range(padding):
        mirror_hsi[i, :, :] = mirror_hsi[padding * 2 - i - 1, :, :]

    for i in range(padding):
        mirror_hsi[height + padding + i, :, :] = mirror_hsi[height + padding - 1 - i, :, :]

    print("**************************************************")
    print("patch is : {}".format(patch))
    print("mirror_image shape : [{0},{1},{2}]".format(mirror_hsi.shape[0], mirror_hsi.shape[1], mirror_hsi.shape[2]))
    print("**************************************************")
    return mirror_hsi


def gain_neighborhood_pixel(mirror_image, point, i, patch=5):
    x = point[i, 0]
    y = point[i, 1]
    temp_image = mirror_image[x:(x + patch), y:(y + patch), :]
    return temp_image


def mirror_padding_band(x_test, band, band_patch, patch=5):
    padding_size = band_patch - 1
    smaller_padding = padding_size // 2
    bigger_padding = padding_size - smaller_padding

    x_padding_band = np.zeros((x_test.shape[0], patch, patch, band + padding_size),
                              dtype=float)

    x_padding_band[:, :, :, 0:bigger_padding] = x_test[:, :, :, 0:bigger_padding][:, :, :, ::-1]

    x_padding_band[:, :, :, bigger_padding:bigger_padding + band] = x_test

    x_padding_band[:, :, :, bigger_padding + band:bigger_padding + band + smaller_padding] = x_test[:, :, :,
                                                                                             band - smaller_padding:band][
                                                                                             :, :, :, ::-1]

    return x_padding_band


def gain_neighborhood_band_mirror(x, band, band_patch, patch=5):
    x_padding_band = mirror_padding_band(x, band, band_patch, patch)  # [B,w,w,band+padding_size]

    return x_padding_band


def get_data(mirror_image, band, test_point, patch=5, band_patch=3):
    x_test = np.zeros((test_point.shape[0], patch, patch, band), dtype=float)

    for j in range(test_point.shape[0]):
        x_test[j, :, :, :] = gain_neighborhood_pixel(mirror_image, test_point, j, patch)
    print("x_test  shape = {}, type = {}".format(x_test.shape, x_test.dtype))
    print("**************************************************")

    x_test_band = gain_neighborhood_band_mirror(x_test, band, band_patch, patch)
    print("x_test_band  shape = {}, type = {}".format(x_test_band.shape, x_test_band.dtype))
    print("**************************************************")
    return x_test_band


def get_label(number_test, num_classes):
    y_test = []
    for i in range(num_classes):
        for k in range(number_test[i]):
            y_test.append(i)

    y_test = np.array(y_test)
    print("y_test: shape = {} ,type = {}".format(y_test.shape, y_test.dtype))
    print("**************************************************")
    return y_test


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, target, pred.squeeze()


def test_epoch(model, valid_loader, criterion):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    sum = 0
    for batch_idx, (batch_data, batch_target) in enumerate(valid_loader):
        batch_data = batch_data.to(device)
        batch_target = batch_target.to(device)
        tic = time.time()
        batch_pred = model(batch_data)
        toc = time.time()
        sum += toc - tic
        loss = criterion(batch_pred, batch_target)

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))

        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    print("time={}".format(sum))
    return tar, pre


def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA


def save_matrix(tar, pre, dataset):
    matrix = confusion_matrix(tar, pre)
    np.save('./confusionMatrix/{}.npy'.format(dataset), matrix)



def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=np.float64)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA


def modify_data(x, patch_size, band_patch):
    x = torch.nn.functional.unfold(x, (patch_size, band_patch))
    return x


# Parameter Setting
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
cudnn.deterministic = True
cudnn.benchmark = False

TE, TR, input = get_dataset(args.dataset)
image_size, near_band, window_size = get_config(args.dataset)

trb = (TR != 0) * 1
teb = (TE != 0) * 1
temp = trb + teb

label = TR + TE
num_classes = np.max(TR)

input_normalize = np.zeros(input.shape)

for i in range(input.shape[2]):
    input_max = np.max(input[:, :, i])
    input_min = np.min(input[:, :, i])
    input_normalize[:, :, i] = (input[:, :, i] - input_min) / (input_max - input_min)

height, width, band = input.shape
print("height={0},width={1},band={2}".format(height, width, band))

total_pos_test, number_test = chooose_point(TE, num_classes)

mirror_image = mirror_hsi(height, width, band, input_normalize, image_size)

x_test_band = get_data(mirror_image, band, total_pos_test,
                       patch=image_size,
                       band_patch=near_band)

y_test = get_label(number_test, num_classes)

x_test = torch.from_numpy(x_test_band).type(torch.FloatTensor)
y_test = torch.from_numpy(y_test).type(torch.LongTensor)

x_test = modify_data(x_test, image_size, near_band)

Label_test = Data.TensorDataset(x_test, y_test)

label_test_loader = Data.DataLoader(Label_test, batch_size=args.batch_size, shuffle=True, num_workers=16)

model = SwinT(image_size=image_size, near_band=near_band, num_patches=band,
              patch_dim=near_band * image_size ** 2, num_classes=num_classes, band=band, dim=64,
              heads=4, dropout=0.1, emb_dropout=0.1, window_size=window_size,
              )

model = model.to(device)
test_model = "./save_checkpoint/" + args.dataset + ".pt"
model.load_state_dict(
    torch.load(test_model, map_location=None if torch.cuda.is_available() else "cpu"))  # cpu

model.eval()
criterion = nn.CrossEntropyLoss().to(device)

tar_v, pre_v = test_epoch(model, label_test_loader, criterion)
OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)
save_matrix(tar_v, pre_v, args.dataset)
print("Final result:")
print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA2, AA_mean2, Kappa2))
print(AA2)
print("**************************************************")
