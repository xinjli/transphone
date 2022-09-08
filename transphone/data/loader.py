from transphone.data.utils import pad_list
import torch
from torch.utils.data import DataLoader


def collate(xy_lst):
    x_lst = [xy[0] for xy in xy_lst]
    y_lst = [xy[1] for xy in xy_lst]

    x = pad_list(x_lst)
    y = pad_list(y_lst)

    return x,y

def read_loader(dataset, batch_size=32):

    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, collate_fn=collate)
    return loader