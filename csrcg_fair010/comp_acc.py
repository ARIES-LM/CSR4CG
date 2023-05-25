
import sys

pred_file = sys.argv[1]
gold_file = sys.argv[2]
result = []
with open(pred_file) as fp, open(gold_file) as fg:
    for lp, lg in zip(fp, fg):
        if lp == lg:
            result.append(1)
        else:
            result.append(0)
print('acc', sum(result)/len(result))

