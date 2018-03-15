import torch
import pickle
import numpy as np
#import copy
import torchvision.models as models
from model.roi_align import RoIAlign
from model.generate_proposals import GenerateProposals
from utils.utils import isnan,infbreak,printmax

from torch.autograd import Variable
from torch.nn.functional import cross_entropy

def smooth_L1(pred,targets,alpha_in,alpha_out,beta=1.0):
    x=(pred-targets)*alpha_in
    xabs=torch.abs(x)
    y1=0.5*x**2/beta
    y2=xabs-0.5*beta
    case1=torch.le(xabs,beta).float()
    case2=1-case1
    return torch.sum((y1*case1+y2*case2)*alpha_out)/pred.size(0)

def accuracy(cls_score,cls_labels):
    class_dim = cls_score.dim()-1
    argmax=torch.max(torch.nn.functional.softmax(cls_score,dim=class_dim),class_dim)[1]
    accuracy = torch.mean(torch.eq(argmax,cls_labels.long()).float())
    return accuracy

# class detector_loss(torch.nn.Module):
#     def __init__(self, do_loss_cls=True, do_loss_bbox=True, do_accuracy_cls=True):
#         super(detector_loss, self).__init__()
#         # Flags
#         self.do_loss_cls = do_loss_cls
#         self.do_loss_bbox = do_loss_bbox
#         self.do_accuracy_cls = do_accuracy_cls
#         # Dicts for losses 
#         # self.losses={}
#         # if do_loss_cls:
#         #     self.losses['loss_cls']=0
#         # if do_loss_bbox:
#         #     self.losses['loss_bbox']=0
#         # # Dicts for metrics       
#         # self.metrics={}
#         # if do_accuracy_cls:
#         #     self.metrics['accuracy_cls']=0

#     def forward(self,
#             cls_score,
#             cls_labels,
#             bbox_pred,
#             bbox_targets,
#             bbox_inside_weights,
#             bbox_outside_weights):

#         # compute losses
#         losses=[]
#         if self.do_loss_cls:
#             loss_cls = cross_entropy(cls_score,cls_labels.long())
#             losses.append(loss_cls)
#         if self.do_loss_bbox:
#             loss_bbox = smooth_L1(bbox_pred,bbox_targets,bbox_inside_weights,bbox_outside_weights)
#             losses.append(loss_bbox)

#         # # compute metrics
#         # if self.do_accuracy_cls:
#         #     self.metrics['accuracy_cls'] = accuracy(cls_score,cls_labels.long())

#         # sum total loss
#         #loss = torch.sum(torch.cat(tuple([v.unsqueeze(0) for v in losses]),0))        

#         # loss.register_hook(printmax)

#         return tuple(losses)
        