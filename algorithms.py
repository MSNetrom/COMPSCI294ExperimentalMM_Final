from typing import List
import math
import torch

def alg_10(data: torch.Tensor, labels: torch.Tensor):
    
    tresholds = 0
    table = []

    for row in range(len(data)):
        table.append((data[row].sum().item(), labels[row].item()))

    sorted_table = sorted(table, key=lambda x: x[0])

    my_class = 0

    for row in range(len(data)):
        #print(sorted_table[row][1])
        if not sorted_table[row][1] == my_class:
            my_class = sorted_table[row][1]
            tresholds += 1

    return math.log2(tresholds + 1)

def alg_11(data: torch.Tensor, labels: torch.Tensor, sizes: List = [5, 10, 20, 40, 100]):

    mecs = []

    for s in sizes:
        p = int(s*len(data)*0.01)
        mecs.append(alg_10(data[:p], labels[:p]))

    return mecs
    