import torch
import torch.nn as nn
import torch.nn.functional as F


class OpticalFilter(nn.Module):
    """
    Optical Filter Module:
    Applies a learnable spectral transmission filter to input spectral radiance.

    Input shape: [B, H, W, C]
    Where C is the number of spectral channels.
    """
    def __init__(self, num_channels=31, start_wl=400, end_wl=700):
        super(OpticalFilter, self).__init__()
        
        # Create a spectral axis
        self.num_channels = num_channels
        # Initialize a 0-1 uniformly random filter
        init_filter = torch.rand(num_channels)

        # Make it a learnable parameter
        self.filter = nn.Parameter(init_filter, requires_grad=True)

    def forward(self, spectral_radiance):
        # spectral_radiance: [B, H, W, C]
        # Apply element-wise multiplication
        return spectral_radiance * self.filter


class STEQuantizerFunction(torch.autograd.Function):
    """
    Straight-Through Estimator for quantization.
    Forward: quantize using step size
    Backward: pass gradients through as if identity
    """
    @staticmethod
    def forward(ctx, x):
        # Quantize x using step size
        quantized = torch.floor(x)
        return quantized

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through: just pass gradients as is
        # if grad_output is None:
        # print(grad_output)
        return grad_output


class ADC(nn.Module):
    """
    ADC Module:
    Learn a quantization step size and quantize the input.
    
    Input shape: [B, H, W]
    """
    def __init__(self, init_step=0.5):
        super(ADC, self).__init__()
        # step_size as learnable parameter
        self.step_size = nn.Parameter(torch.tensor(init_step), requires_grad=True)

    def forward(self, x):
        # x: [B, H, W]
        x = x / self.step_size
        x = STEQuantizerFunction.apply(x)
        # print("hi")
        return x 


class Demosaic(nn.Module):
    """
    Demosaic Module:
    Assume a simple RGGB Bayer pattern and 2x2 kernel weights.
    We'll treat demosaic as learnable 2x2 convolution kernels for each output channel.
    
    Input shape: [B, H, W]
    Bayer pattern (RGGB):
      R G
      G B
    
    We will model demosaic as learning 3 sets of 2x2 kernels:
    - one for R
    - one for G
    - one for B
    
    We'll then apply them using a stride of 2 to reconstruct a reduced-resolution image
    which is then upsampled to the original size.
    """
    def __init__(self):
        super(Demosaic, self).__init__()
        # We have 3 channels out (R,G,B), each from a 2x2 region.
        # Initialize kernels as described:
        # R: upper-left pixel = 1
        # G: average of upper-right and bottom-left = 0.5, 0.5
        # B: bottom-right pixel = 1
        #
        # We'll store as a single conv weight [out_channels=3, in_channels=1, kernel=2x2]
        
        init_kernel = torch.zeros(3, 1, 2, 2)
        # R channel
        init_kernel[0,0,0,0] = 1.0
        # G channel
        init_kernel[1,0,0,1] = 0.5
        init_kernel[1,0,1,0] = 0.5
        # B channel
        init_kernel[2,0,1,1] = 1.0

        self.kernel = nn.Parameter(init_kernel, requires_grad=True)

    def forward(self, mosaic):
        # mosaic: [B, H, W]
        # Add channel dimension to mosaic
        mosaic = mosaic.unsqueeze(1)  # [B,1,H,W]

        # Apply conv with stride=2
        out = F.conv2d(mosaic, self.kernel, stride=2) # [B, 3, H/2, W/2]

        # Upsample back to original size
        # Assuming bilinear upsampling
        H, W = mosaic.shape[2], mosaic.shape[3]
        out_upsampled = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out_upsampled
