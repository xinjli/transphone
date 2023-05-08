from pathlib import Path
from transphone.config import TransphoneConfig
import shutil
import yaml


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def read_model_config(exp):
    yaml_file = TransphoneConfig.data_path / 'exp' / f'{exp}.yml'
    # Open the YAML file and load its contents into a Python dictionary
    with open(yaml_file, "r") as f:
        model_config = yaml.safe_load(f)

    return dotdict(model_config)


def resolve_model_name(model_name='latest', alt_model_path=None):
    """
    select the model

    :param model_name:
    :return:
    """

    models = {
        'latest': '042801_base',
        '042801_base': '042801_base'
    }

    assert model_name in models, f"{model_name} is not available"

    return models[model_name]


def pad_list(tensor_lst):
    max_length = max(t.size(0) for t in tensor_lst)
    batch_size = len(tensor_lst)

    padded_tensor = tensor_lst[0].new(batch_size, max_length, *tensor_lst[0].size()[1:]).fill_(0)
    for i, t in enumerate(tensor_lst):
        padded_tensor[i,:t.size(0)] = t

    return padded_tensor


def pad_sos_eos(ys, sos, eos):

    batch_size = len(ys)

    sos_tensor = ys.new_zeros((batch_size, 1)).fill_(sos)
    y_in = torch.cat([sos_tensor, ys], dim=1)

    y_out = []

    zero_tensor = ys.new_zeros((batch_size, 1))
    extended_ys = torch.cat([ys, zero_tensor], dim=1)

    for y in extended_ys:
        #eos_idx = (y==0).nonzero()[0].item()
        #print(y)
        eos_idx = torch.nonzero(y==0)[0].item()
        y[eos_idx] = eos
        y_out.append(y)

    y_out = pad_list(y_out)
    return y_in, y_out