import torch
from typing import Tuple

def conv_out_size(in_size: Tuple[int, int, int], kernel_size: Tuple[int, int], stride: Tuple[int, int], padding: Tuple[int, int], filters: int) -> Tuple[int, int, int]:
    # in_size: (channels, height, width)

    out_channels = filters
    out_height = (in_size[1] - kernel_size[0] + 2*padding[0]) // stride[0] + 1
    out_width = (in_size[2] - kernel_size[1] + 2*padding[1]) // stride[1] + 1

    return (out_channels, out_height, out_width)

def full_cnn_seq_size(cnn_sequence: torch.nn.Sequential, in_size: Tuple[int, int, int]):

    # in_size: (channels, height, width)
    sizes = [in_size]

    for layer in cnn_sequence:
        if isinstance(layer, torch.nn.Conv2d):
            sizes.append(conv_out_size(sizes[-1], layer.kernel_size, layer.stride, layer.padding, layer.out_channels))
        elif isinstance(layer, torch.nn.MaxPool2d):
            sizes.append(conv_out_size(sizes[-1], (layer.kernel_size, layer.kernel_size), (layer.stride, layer.stride), (layer.padding, layer.padding), sizes[-1][0]))

    return sizes[1:]

def cnn_layer_mec_no_cap(cnn_layer: torch.nn.Conv2d) -> int:

    return (cnn_layer.in_channels * cnn_layer.kernel_size[0] * cnn_layer.kernel_size[1] + 1) * cnn_layer.out_channels

def cnn_mec_calc(cnn_sequence: torch.nn.Sequential, in_size: Tuple[int, int, int]) -> int:

    cnn_sequence = list(filter(lambda x: isinstance(x, torch.nn.Conv2d) or isinstance(x, torch.nn.MaxPool2d), cnn_sequence))

    out_sizes = full_cnn_seq_size(cnn_sequence, in_size)

    total_mec = cnn_layer_mec_no_cap(cnn_sequence[0])

    #print("Layer 0:", cnn_sequence[0], "No cap MEC:", cnn_layer_mec_no_cap(cnn_sequence[0]))

    for layer, size in zip(cnn_sequence[1:], out_sizes):
        if isinstance(layer, torch.nn.Conv2d):
            #print("Size:", size)
            #print("Layer:", layer, "Size:", size, "No cap MEC:", cnn_layer_mec_no_cap(layer), "Cap:", size[0] * size[1] * size[2])
            total_mec += min(cnn_layer_mec_no_cap(layer), size[0] * size[1] * size[2])

    return total_mec



