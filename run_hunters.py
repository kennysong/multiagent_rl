import numpy as np
import os
import random
import sys
import torch

from namedlist import namedlist
EpisodeStep = namedlist('EpisodeStep', 's a grad_W r G', default=0)

episode = torch.load('episode')

for step in episode:
    s, a, r = step.s, step.a, step.r

    print('State: ' + str(s))
    print('Action: ' + str(a))
    print('Reward: ' + str(r))

    for row in range(6):
        rowstr = ''
        for col in range(6):
            if (s[1] == row and s[2] == col) or \
               (s[4] == row and s[5] == col):
                rowstr += 'H '
            elif (s[7] == row and s[8] == col) or \
                 (s[10] == row and s[11] == col):
                rowstr += 'R '
            else:
                rowstr += '. '
        print(rowstr)
    raw_input()

# 0 1 2 | 3 4 5 | 6 7 8 | 9 0 1
