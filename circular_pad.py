#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-27

from torch import tensor
from torch.autograd import Function
import collections
from itertools import repeat
import torch.nn as nn


        
class CircularPad2d_Function(Function):    
    @staticmethod
    def forward(self, input, padding):
        assert input.dim() == 4, "only 4D supported for padding"
        
        #global padding, input_size
        
        pad_l, pad_r, pad_t, pad_b = padding
        h = input.size(2) + pad_t + pad_b
        w = input.size(3) + pad_l + pad_r
        assert w > 0 and h > 0, "input is too small"

        input_size = input.size()
        self.save_for_backward(input, tensor(padding))
        
        
        output = input.new(input.size(0), input.size(1), h, w).zero_()

        # crop output if necessary
        c_output = output

        if pad_t > 0:
            c_output = c_output.narrow(2, pad_t, c_output.size(2) - pad_t)
        if pad_b > 0:
            c_output = c_output.narrow(2, 0, c_output.size(2) - pad_b)

        # circular padding
        c_output[:, :, :, 0:pad_l] = input[:, :, :, -pad_r:]
        c_output[:, :, :, -pad_r:] = input[:, :, :, 0:pad_l]

        if pad_l > 0:
            c_output = c_output.narrow(3, pad_l, c_output.size(3) - pad_l)
        if pad_r > 0:
            c_output = c_output.narrow(3, 0, c_output.size(3) - pad_r)
        c_output.copy_(input)

        return output
    
    @staticmethod
    def backward(self, grad_output):
        
        input, padding = self.saved_tensors
        input_size = input.size()
        
        pad_l, pad_r, pad_t, pad_b = padding[0], padding[1], padding[2], padding[3]

        grad_input = grad_output.new(input_size).zero_()

        cg_input = grad_input

        # crop grad_output if necessary
        cg_output = grad_output
        if pad_t > 0:
            cg_output = cg_output.narrow(2, pad_t, cg_output.size(2) - pad_t)
        if pad_b > 0:
            cg_output = cg_output.narrow(2, 0, cg_output.size(2) - pad_b)
        if pad_l > 0:
            cg_output = cg_output.narrow(3, pad_l, cg_output.size(3) - pad_l)
        if pad_r > 0:
            cg_output = cg_output.narrow(3, 0, cg_output.size(3) - pad_r)
        cg_input.copy_(cg_output)

        cg_input[:, :, :, 0:pad_l] += grad_output[
            :, :, pad_t : grad_output.size(2) - pad_b, -pad_r:
        ]
        cg_input[:, :, :, -pad_r:] += grad_output[
            :, :, pad_t : grad_output.size(2) - pad_b, 0:pad_l
        ]

        return grad_input, None
    
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable): #collections.Iterable
            return x
        return tuple(repeat(x, n))

    return parse


_quadruple = _ntuple(4)

class CircularPad2d(nn.Module):
    def __init__(self, padding=(1, 1, 1, 1)):
        super(CircularPad2d, self).__init__()
        self.padding = _quadruple(padding)

    def forward(self, input):
        x = CircularPad2d_Function.apply(input, self.padding)
        return x

    def __repr__(self):
        return self.__class__.__name__ + str(self.padding)