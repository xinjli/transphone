# epitran evaluation

This contains the epitran evaluation based on supported epitran performance. These are tested on the same testing set as the supervised model.

Note that it might not be fair to compare epitran with our model as our model is trained using the Wikitionary training set and epitran is trained using other sources. so our model's training set is consistent with the testing set, but epitran's training set is not. This is only included for reference.

This can be reproduced by `transphone.bin.eval_epitran`

| language | phoneme error rate | phonological distance |
|----------|--------------------|-----------------------|
| tur | 0.045 | 0.032 |
| vie | 0.308 | 0.151 |
| aar | 0.222 | 0.026 |
| got | 0.851 | 0.391 |
| swa | 0.036 | 0.036 |
| swe | 0.263 | 0.050 |
| amh | 0.239 | 0.112 |
| hat | 0.025 | 0.004 |
| tam | 0.198 | 0.110 |
| mal | 0.357 | 0.106 |
| hin | 0.317 | 0.080 |
| mar | 0.202 | 0.056 |
| tel | 0.467 | 0.199 |
| ara | 0.359 | 0.218 |
| mlt | 0.169 | 0.053 |
| tgk | 0.074 | 0.044 |
| mon | 0.403 | 0.145 |
| tgl | 0.283 | 0.081 |
| hun | 0.030 | 0.011 |
| msa | 0.178 | 0.025 |
| tha | 0.208 | 0.072 |
| ben | 0.217 | 0.082 |
| ilo | 0.199 | 0.092 |
| mya | 0.239 | 0.136 |
| ind | 0.125 | 0.017 |
| nan | 0.373 | 0.373 |
| ita | 0.152 | 0.042 |
| jam | 0.438 | 0.208 |
| nld | 0.256 | 0.075 |
| cat | 0.292 | 0.057 |
| nya | 0.142 | 0.135 |
| tuk | 0.144 | 0.029 |
| ceb | 0.095 | 0.067 |
| ces | 0.338 | 0.017 |
| ori | 0.064 | 0.036 |
| ckb | 0.297 | 0.112 |
| kat | 0.005 | 0.000 |
| pan | 0.857 | 0.383 |
| kaz | 0.985 | 0.369 |
| uig | 0.276 | 0.033 |
| csb | 0.259 | 0.159 |
| ukr | 0.368 | 0.064 |
| deu | 0.335 | 0.071 |
| pol | 0.050 | 0.046 |
| urd | 0.522 | 0.328 |
| kbd | 0.195 | 0.129 |
| por | 0.427 | 0.081 |
| ron | 0.023 | 0.002 |
| fas | 0.398 | 0.255 |
| kir | 1.029 | 0.402 |
| rus | 0.255 | 0.061 |
| xho | 0.418 | 0.120 |
| yor | 0.092 | 0.059 |
| kmr | 0.050 | 0.009 |
| yue | 0.197 | 0.043 |
| zha | 0.079 | 0.021 |
| lao | 0.419 | 0.260 |
| spa | 0.058 | 0.043 |
| zul | 0.090 | 0.036 |
| sqi | 0.041 | 0.017 |
| lij | 0.767 | 0.173 |