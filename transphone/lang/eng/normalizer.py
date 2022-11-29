"""
This code is from: https://github.com/tomaarsen/TTSTextNormalization/blob/master/converters/Digit.py

"""

import re

class ENGNormalizer:

    def __init__(self):
        super().__init__()
        # Regex used to filter out non digits
        self.filter_regex = re.compile("[^0-9]")
        # Translation dict to convert digits to text
        self.trans_dict = {
            "0": "zero",
            "1": "one",
            "2": "two",
            "3": "three",
            "4": "four",
            "5": "five",
            "6": "six",
            "7": "seven",
            "8": "eight",
            "9": "nine"
        }

        self.punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'


    def __call__(self, text: str) -> str:

        text = text.translate(str.maketrans('', '', self.punctuation)).lower()

        tokens = text.split()

        res = []
        for token in tokens:
            if str.isdigit(token):
                # 1 Filter out anything that isn't a digit
                token = self.filter_regex.sub("", text)
                # 2 Check for special case
                if token == "007":
                    return "double o seven"
                # 3 & 4 Convert each digit to text and space out the text
                token = " ".join([self.trans_dict[c] for c in token])

            res.append(token)

        return " ".join(res)
