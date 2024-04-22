import torch
import torch.nn as nn


# basic swish
class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


# Swish with learnable parameter
class SwishParam(nn.Module):
    def __init__(self):
        super(SwishParam, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))  # or nn.Parameter(torch.ones(1))?

    def forward(self, x):
        return x * torch.sigmoid(x * self.beta)


# Gated Linear Unit: GLU(x, W, V, b, c) = \sigma(xW+b) x (xV+c)
class GatedLinearUnit(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GatedLinearUnit, self).__init__()
        # Linear transformation for the gate
        self.W = nn.Linear(input_dim, output_dim)
        # Linear transformation for the value
        self.V = nn.Linear(input_dim, output_dim)
        # Biases are included in nn.Linear by default

    def forward(self, x):
        # Apply linear transformations
        gate = torch.sigmoid(self.W(x))
        value = self.V(x)
        # Return the element-wise product
        return gate * value


# Swish-GLU
class SwishGLU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SwishGLU, self).__init__()
        # Linear transformation for the gate
        self.W = nn.Linear(input_dim, output_dim)
        # Linear transformation for the value
        self.V = nn.Linear(input_dim, output_dim)
        self.swish = SwishParam()
        # Biases are included in nn.Linear by default

    def forward(self, x):
        # Apply linear transformations
        gate = self.swish(self.W(x))
        value = self.V(x)
        # Return the element-wise product
        return gate * value


'''
The Data Adaptive Activation Function in DIN, which can be viewed as a generalization of PReLu
and can adaptively adjust the rectified point according to distribution of input data

Refer to the PRelu, the key idea of Dice is to adaptively adjust the rectified point according to distribution of 
input data, whose value is set to be the mean of input.
'''


class Dice(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        super(Dice, self).__init__()
        self.alpha = nn.Parameter(torch.zeros(1))  # Trainable parameter, can be multiple
        self.bn = nn.BatchNorm1d(num_features, eps=eps)  # Batch normalization

    def forward(self, x):
        # Apply batch normalization
        x_norm = self.bn(x)

        # Sigmoid activation function to get p(x)
        px = torch.sigmoid(x_norm)

        # Dice activation
        y = px * x + (1 - px) * self.alpha * x
        return y
