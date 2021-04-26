# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Miscellaneous utility functions
"""

import torch


def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


if __name__ == '__main__':
    x = [[1, 2, 3, 4],
         [1, 2, 3, 4]]
    x = torch.tensor(x)
    y = [[1, 2, 3, 4],
         [1, 2, 3, 4]]
    y = torch.tensor(y)

    h = [x, y]

    h = torch.cat(h, dim=0)
    print("the shape of h:", h.shape)
    print("h:", h)
