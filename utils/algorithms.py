from typing import List
import math
import torch

def alg_8(data: torch.Tensor, labels: torch.Tensor):
    d = data.shape[1]  # Assuming data is a 2D tensor where the second dimension is the number of features
    thresholds = 0
    table = []

    # Populate the table with sums of rows and their corresponding labels
    for row in range(len(data)):
        table.append((data[row].sum().item(), labels[row].item()))

    # Sort the table by the first column (sum of rows)
    sorted_table = sorted(table, key=lambda x: x[0])

    current_class = 0

    # Identify the number of thresholds required
    for row in range(len(sorted_table)):
        if sorted_table[row][1] != current_class:
            current_class = sorted_table[row][1]
            thresholds += 1

    # Compute minthreshs as log2(thresholds + 1)
    minthreshs = math.log2(thresholds + 1)

    # Compute MEC (memory-equivalent capacity)
    mec = (minthreshs * (d + 1)) + (minthreshs + 1)

    return mec

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
    