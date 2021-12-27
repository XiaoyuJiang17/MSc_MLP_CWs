from model_architectures import *
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# Test whether all the output of BN layer has mean 0, variance 1 and Residual Connection Correctly implemented in Processing block

# Initialize a Test block
input_shape = (100,3,32,32)
num_filters = 3
kernel_size = 3
padding, bias = 1, False
dilation = 1

all_correct = True

Test_block = ConvolutionalProcessingBlock_BNRC(input_shape, num_filters, kernel_size, padding, bias, dilation)
Test_input = torch.randn(input_shape)

# Test for shape
Test_block_output = Test_block.forward(Test_input)
if not Test_block_output.shape == Test_input.shape:
    all_correct = False
    print('This is NOT a processing block, check if there is a mistake')

# Test for BN output properity
out = Test_block.layer_dict['conv_0'].forward(Test_input)
out = Test_block.layer_dict['bn_0'].forward(out)
if not np.allclose(out.mean().detach().numpy(),0):
    all_correct = False
    print('bn_0 does not have zero mean output')
elif not ( out.std().detach().numpy() > 0.999 and out.std().detach().numpy() < 1.001 ):
    all_correct = False
    print('bn_0 does not have unit(one) std output')

out = F.leaky_relu(out)
out = Test_block.layer_dict['conv_1'].forward(out)
out = Test_block.layer_dict['bn_1'].forward(out)
if not np.allclose(out.mean().detach().numpy(),0):
    all_correct = False
    print('bn_1 does not have zero mean output')
elif not ( out.std().detach().numpy() > 0.999 and out.std().detach().numpy() < 1.001 ):
    all_correct = False
    print('bn_1 does not have unit(one) std output')

# Test for Residual Connection

if not torch.all(torch.eq(Test_block_output, F.leaky_relu(out + Test_input))):
    print('The Residual Connection NOT correctly implemented')

if all_correct:
    print('All Tests passes')