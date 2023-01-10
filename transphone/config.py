from pathlib import Path
import logging
import torch

class TransphoneConfig:

    root_path = Path(__file__).parent.parent
    data_path = Path(__file__).parent / 'pretrained'
    lang_path = data_path / 'lang'
    logger = logging.getLogger('transphone')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
