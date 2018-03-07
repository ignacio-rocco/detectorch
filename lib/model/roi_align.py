import torch
from torch.autograd import Function
from torch.nn.modules.module import Module
from torch.autograd import Variable 
import os

torch_ver = torch.__version__[:3]

if torch_ver=="0.4":
    from torch.utils.cpp_extension import load   
    build_path = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../cppcuda/build/'))

    print('compiling/loading roi_align')
    roialign = load(name='roialign',sources=['lib/cppcuda/roi_align_binding.cpp',
                                            'lib/cppcuda/roi_align_forward_cuda.cu',
                                            'lib/cppcuda/roi_align_backward_cuda.cu'],
                    build_directory=build_path,verbose=True)
else:
    import cppcuda_cffi.roialign as roialign


class RoIAlignFunction(Function):
    def __init__(self, pooled_height, pooled_width, spatial_scale, sampling_ratio):
        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = int(sampling_ratio)
        self.features_size = None

    def forward(self, features, rois):
        # compute
        if features.is_cuda != rois.is_cuda:
            raise TypeError('features and rois should be on same device (CPU or GPU)')
        elif features.is_cuda and rois.is_cuda :
            if torch_ver=="0.4":
                output = roialign.roi_align_forward_cuda(features,
                                                rois,
                                                self.pooled_height,
                                                self.pooled_width,
                                                self.spatial_scale,
                                                self.sampling_ratio)
            else:
                num_channels = features.size(1)
                num_rois = rois.size(0)
                output = torch.zeros(num_rois, num_channels, self.pooled_height, self.pooled_width).cuda()
                roialign.roi_align_forward_cuda(features,
                                rois,
                                output,
                                self.pooled_height,
                                self.pooled_width,
                                self.spatial_scale,
                                self.sampling_ratio)
        elif features.is_cuda==False and rois.is_cuda==False:
            if torch_ver=="0.4":
                output = roialign.roi_align_forward_cpu(features,
                                                rois,
                                                self.pooled_height,
                                                self.pooled_width,
                                                self.spatial_scale,
                                                self.sampling_ratio)
            else:
                num_channels = features.size(1)
                num_rois = rois.size(0)
                output = torch.zeros(num_rois, num_channels, self.pooled_height, self.pooled_width)
                roialign.roi_align_forward_cpu(features,
                                rois,
                                output,
                                self.pooled_height,
                                self.pooled_width,
                                self.spatial_scale,
                                self.sampling_ratio)
      
        self.features_size = features.size()
        self.save_for_backward(rois)

        if torch_ver=="0.4":
            return Variable(output,requires_grad=True)
        else:
            return output
 

    def backward(self, grad_output):
        rois = self.saved_variables[0]
        if rois.is_cuda:
            if torch_ver=="0.4":
                grad_input = roialign.roi_align_backward_cuda(rois,
                                    grad_output,
                                    self.features_size[0],
                                    self.features_size[1],
                                    self.features_size[2],
                                    self.features_size[3],
                                    self.pooled_height,
                                    self.pooled_width,
                                    self.spatial_scale,
                                    self.sampling_ratio)
            else:
                grad_input = torch.zeros(self.features_size).cuda()
                roialign.roi_align_backward_cuda(rois,
                                grad_output,
                                grad_input,
                                self.pooled_height,
                                self.pooled_width,
                                self.spatial_scale,
                                self.sampling_ratio)
            
        else:
            if torch_ver=="0.4":
                grad_input = roialign.roi_align_backward_cpu(rois,
                                    grad_output,
                                    self.features_size[0],
                                    self.features_size[1],
                                    self.features_size[2],
                                    self.features_size[3],
                                    self.pooled_height,
                                    self.pooled_width,
                                    self.spatial_scale,
                                    self.sampling_ratio)
            else:
                raise("backward pass not implemented on cpu in cffi extension")

        if torch_ver=="0.4":
            return Variable(grad_input), None
        else:
            return grad_input




class RoIAlign(Module):
    def __init__(self, pooled_height, pooled_width, spatial_scale, sampling_ratio=0):
        super(RoIAlign, self).__init__()

        self.roialignfun = RoIAlignFunction(int(pooled_height),
                                            int(pooled_width),
                                            float(spatial_scale),
                                            int(sampling_ratio))

    def forward(self, features, rois):
        # features is a Variable/FloatTensor of size BxCxHxW
        # rois is a (optional: list of) Variable/FloatTensor IDX,Xmin,Ymin,Xmax,Ymax (normalized to [0,1])
        rois = self.preprocess_rois(rois)
        output = self.roialignfun(features,rois)
        return output
       

    def preprocess_rois(self, rois):
        # do some verifications on what has been passed as rois
        if isinstance(rois,list): # if list, convert to single tensor (used for multiscale)
            rois = torch.cat(tuple(rois),0)
        if isinstance(rois,Variable):
            if rois.dim()==3:
                if rois.size(0)==1:
                    rois = rois.squeeze(0)
                else:
                    raise("rois has wrong size")
            if rois.size(1)==4:
                # add zeros
                zeros = Variable(torch.zeros((rois.size(0),1)))
                if rois.is_cuda:
                    zeros = zeros.cuda()
                rois = torch.cat((zeros,rois),1).contiguous()
        return rois