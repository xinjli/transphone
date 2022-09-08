from pathlib import Path


class TransphoneConfig:

    root_path = Path(__file__).parent.parent
    data_path = Path(__file__).parent / 'pretrained'
    lang_path = data_path / 'lang'
