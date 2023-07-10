#

#generate token-dataidx
import sys

token_dataidx = {}
for i, l in enumerate(open(sys.argv[1], encoding="utf-8")):
    for w in l.strip().split():
        if w not in token_dataidx:
            token_dataidx[w] = [i]
        else:
            token_dataidx[w].append(i)

import pickle

write2 = sys.argv[2]

with open(write2, 'wb') as fw:
    pickle.dump(token_dataidx, fw)

