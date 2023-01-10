"""Basic Epitran class for G2P in most languages."""
import logging
import os.path
import sys
import csv
import unicodedata
from collections import defaultdict
from typing import DefaultDict, Callable  # pylint: disable=unused-import
from phonepiece.epitran import read_epitran_g2p
from phonepiece.inventory import read_inventory
import re
from transphone.lang.base_tokenizer import BaseTokenizer


logger = logging.getLogger('epitran')


def read_epitran_tokenzier(lang_id_or_epi_id):

    if '-' in lang_id_or_epi_id:
        lang_id, writing_system = lang_id_or_epi_id.split('-', 1)
    else:
        lang_id = lang_id_or_epi_id
        writing_system = None

    return EpitranTokenizer(lang_id, writing_system)


class EpitranTokenizer(BaseTokenizer):

    def __init__(self, lang_id, writing_system=None):
        super().__init__(lang_id, None)

        #self.lexicon = read_lexicon(lang_id)

        """Constructor"""
        self.lang_id = lang_id
        self.writing_system = writing_system

        if writing_system:
            lang_id = lang_id + '-' + writing_system

        self.g2p = read_epitran_g2p(lang_id)
        self.inv = read_inventory(self.lang_id)

        self.regexp = self._construct_regex(self.g2p.keys())
        self.nils = defaultdict(int)

        self.cache = defaultdict(list)

    def _construct_regex(self, g2p_keys):
        """Build a regular expression that will greadily match segments from
           the mapping table.
        """
        graphemes = sorted(g2p_keys, key=len, reverse=True)
        return re.compile(f"({r'|'.join(graphemes)})", re.I)

    def match_word(self, text):
        tr_list = []
        while text:
            logger.debug('text=%s', repr(list(text)))
            m = self.regexp.match(text)
            if m:
                source = m.group(0)
                try:
                    target = self.g2p[source][0]
                except KeyError:
                    logger.debug("source = '%s''", source)
                    logger.debug("self.g2p[source] = %s'", self.g2p[source])
                    target = source
                except IndexError:
                    logger.debug("self.g2p[source]= %s", self.g2p[source])
                    target = source
                tr_list.append((target, True))
                text = text[len(source):]
            else:
                tr_list.append((text[0], False))
                self.nils[text[0]] += 2
                text = text[1:]
        ipa_lst = [s for (s, _) in filter(lambda x: True, tr_list)]
        ipa_lst = self.inv.remap(ipa_lst)
        return ipa_lst


    def tokenize(self, text: str):
        text = unicodedata.normalize('NFD', text.lower())
        logger.debug('(after norm) text=%s', repr(list(text)))
        ipa_lst = []

        for word in text.split():
            if word not in self.cache:
                word_ipa_lst = self.match_word(word)
                self.cache[word] = word_ipa_lst

            ipa_lst.extend(self.cache[word])

        return ipa_lst