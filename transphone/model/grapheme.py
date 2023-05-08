from phonepiece.unit import Unit
from pathlib import Path
from phonepiece.config import phonepiece_config
from unidecode import unidecode
import editdistance

def read_grapheme(lang_id):

    unit_to_id = dict()

    unit_to_id['<blk>'] = 0

    idx = 0

    unit_path = Path(phonepiece_config.data_path / 'phonetisaurus' / lang_id / 'char.txt')

    for line in open(str(unit_path), 'r', encoding='utf-8'):
        fields = line.strip().split()

        assert len(fields) < 3

        if len(fields) == 1:
            unit = fields[0]
            idx += 1
        else:
            unit = fields[0]
            idx = int(fields[1])

        unit_to_id[unit] = idx

    unit_to_id['<blk>'] = 0
    unit_to_id['<eos>'] = idx+1

    return Grapheme(unit_to_id)


class Grapheme(Unit):

    def __init__(self, unit_to_id):
        super().__init__(unit_to_id)

        self.latins = []
        self.nearest_mapping = None

        for i, elem in enumerate(self.elems):
            if i == 0:
                self.latins.append('<blk>')
            else:
                self.latins.append(unidecode(elem))

    def get_nearest_unit(self, unit):

        if self.nearest_mapping is None:
            self.nearest_mapping = dict()

        if unit in self.nearest_mapping:
            return self.nearest_mapping[unit]

        if unit in self.unit_to_id:
            self.nearest_mapping[unit] = unit
            return unit

        target_latin = unidecode(unit)

        edit_score = dict()
        for i, latin in enumerate(self.latins):
            unit = self.id_to_unit[i]
            edit_score[unit] = editdistance.eval(latin, target_latin)

        target_unit = min(edit_score, key=edit_score.get)
        self.nearest_mapping[unit] = target_unit

        return target_unit