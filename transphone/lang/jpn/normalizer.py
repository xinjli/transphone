# encoding: utf8
"""
from: https://github.com/neologd/mecab-ipadic-neologd/wiki/Regexp.ja
"""
from __future__ import unicode_literals
import re
import unicodedata

def unicode_normalize(cls, s):
    pt = re.compile('([{}]+)'.format(cls))

    def norm(c):
        return unicodedata.normalize('NFKC', c) if pt.match(c) else c

    s = ''.join(norm(x) for x in re.split(pt, s))
    s = re.sub('－', '-', s)
    return s

def remove_extra_spaces(s):
    s = re.sub('[ 　]+', ' ', s)
    blocks = ''.join(('\u4E00-\u9FFF',  # CJK UNIFIED IDEOGRAPHS
                      '\u3040-\u309F',  # HIRAGANA
                      '\u30A0-\u30FF',  # KATAKANA
                      '\u3000-\u303F',  # CJK SYMBOLS AND PUNCTUATION
                      '\uFF00-\uFFEF'   # HALFWIDTH AND FULLWIDTH FORMS
                      ))
    basic_latin = '\u0000-\u007F'

    def remove_space_between(cls1, cls2, s):
        p = re.compile('([{}]) ([{}])'.format(cls1, cls2))
        while p.search(s):
            s = p.sub(r'\1\2', s)
        return s

    s = remove_space_between(blocks, blocks, s)
    s = remove_space_between(blocks, basic_latin, s)
    s = remove_space_between(basic_latin, blocks, s)
    return s

def normalize_neologd(s):
    s = s.strip()
    s = unicode_normalize('０-９Ａ-Ｚａ-ｚ｡-ﾟ', s)

    def maketrans(f, t):
        return {ord(x): ord(y) for x, y in zip(f, t)}

    s = re.sub('[˗֊‐‑‒–⁃⁻₋−]+', '-', s)  # normalize hyphens
    s = re.sub('[﹣－ｰ—―─━ー]+', 'ー', s)  # normalize choonpus
    s = re.sub('[~∼∾〜〰～]', '', s)  # remove tildes
    s = s.translate(
        maketrans('!"#$%&\'()*+,-./:;<=>?@[¥]^_`{|}~｡､･｢｣',
              '！”＃＄％＆’（）＊＋，－．／：；＜＝＞？＠［￥］＾＿｀｛｜｝〜。、・「」'))

    s = remove_extra_spaces(s)
    s = unicode_normalize('！”＃＄％＆’（）＊＋，－．／：；＜＞？＠［￥］＾＿｀｛｜｝〜', s)  # keep ＝,・,「,」
    s = re.sub('[’]', '\'', s)
    s = re.sub('[”]', '"', s)
    return s


def parse_small_jpn_number(num):

    if num == 0:
        return "ぜろ"
    if num == 1:
        return "いち"

    digit = ["", "いち", "に", "さん", "よん", "ご", "ろく", "なな", "はち", "きゅう"]
    unit = ["", "じゅう", "ひゃく", "せん"]
    dakuon_unit = ["", "じゅう", "びゃく", "ぜん"]
    sokuon_unit = ["", "じゅう", "ぴゃく", "せん"]

    num_str = str(num)
    n = len(num_str)
    read_list = []
    for i in range(n):
        d = int(num_str[i])
        dakuon = False
        sokuon = False
        if d != 0:
            # skip 1 unless it is the last char
            if i != n-1 and d == 1:
                read_list.append("")
            # 濁音
            elif d == 3:
                dakuon = True
                read_list.append(digit[d])
            elif d == 6 and n-i == 3:
                read_list.append("ろっ")
                sokuon = True
            elif d == 8 and n-i == 3:
                read_list.append("はっ")
                sokuon = True
            else:
                read_list.append(digit[d])

            if dakuon:
                read_list.append(dakuon_unit[n-i-1])
            elif sokuon:
                read_list.append(sokuon_unit[n-i-1])
            else:
                read_list.append(unit[n - i - 1])
    read_str = "".join(read_list)
    return read_str


def parse_jpn_number(num):

    num_size = len(str(num))
    if num_size <= 4:
        return parse_small_jpn_number(num)
    elif num_size <= 8:
        low_digit = num%10000
        high_digit = num//10000

        high_read = parse_small_jpn_number(high_digit) + 'まん'
        low_read = ""
        if low_digit != 0:
            low_read = parse_small_jpn_number(low_digit)
        return high_read + low_read
    else:

        low_digit = num % 10000
        mid_digit = (num//10000)%10000
        high_digit = num//100000000

        high_read = parse_small_jpn_number(high_digit) + 'おく'
        mid_read = ""
        if mid_digit != 0:
            mid_read = parse_small_jpn_number(mid_digit) + 'まん'
        low_read = ""
        if low_digit != 0:
            low_read = parse_small_jpn_number(low_digit)

        return high_read + mid_read + low_read