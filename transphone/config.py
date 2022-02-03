from pathlib import Path


class TransphoneConfig:

    data_path = Path(__file__).parent.parent / 'data'
    lang_path = data_path / 'lang'