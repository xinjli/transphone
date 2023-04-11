"""
This code is from: https://github.com/tomaarsen/TTSTextNormalization/blob/master/converters/Digit.py

"""

import re


# Function to convert numbers to words, adapted from GPT generation
def number_to_words(num):

    # Create a dictionary for mapping digits to words
    digits = {
        0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four',
        5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine'
    }

    # Create a dictionary for mapping two-digit numbers to words
    two_digits = {
        10: 'ten', 11: 'eleven', 12: 'twelve', 13: 'thirteen', 14: 'fourteen',
        15: 'fifteen', 16: 'sixteen', 17: 'seventeen', 18: 'eighteen', 19: 'nineteen',
        20: 'twenty', 30: 'thirty', 40: 'forty', 50: 'fifty', 60: 'sixty',
        70: 'seventy', 80: 'eighty', 90: 'ninety'
    }

    # Create a dictionary for mapping powers of 10 to words
    powers_of_10 = {
        100: 'hundred', 1000: 'thousand', 1000000: 'million',
        1000000000: 'billion', 1000000000000: 'trillion'
    }

    # Handle negative numbers
    if num < 0:
        return "minus " + number_to_words(abs(num))

    # Handle numbers from 0 to 9
    if num < 10:
        return digits[num]

    # Handle numbers from 10 to 99
    if num < 100:
        if num in two_digits:
            return two_digits[num]
        else:
            return two_digits[num // 10 * 10] + " " + digits[num % 10]

    # Handle numbers from 100 to 999
    if num < 1000:
        hundreds = num // 100
        remainder = num % 100
        if remainder == 0:
            return digits[hundreds] + " " + powers_of_10[100]
        else:
            return digits[hundreds] + " " + powers_of_10[100] + " and " + number_to_words(remainder)

    # Handle numbers greater than or equal to 1000
    for power in sorted(powers_of_10.keys(), reverse=True):
        if num >= power:
            quotient = num // power
            remainder = num % power
            if remainder == 0:
                return number_to_words(quotient) + " " + powers_of_10[power]
            else:
                return number_to_words(quotient) + " " + powers_of_10[power] + " " + number_to_words(remainder)

    return "infinity"

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

                token = number_to_words(int(token))

            res.append(token)

        return " ".join(res)
