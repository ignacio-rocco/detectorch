import torch
import pickle
import numpy as np
#import copy
import torchvision.models as models
from model.roi_align import RoIAlign
from model.generate_proposals import GenerateProposals

class mask_head(torch.nn.Module):
    def __init__(self, conv_head, roi_align, output_prob):
        super(mask_head, self).__init__()
        self.output_prob = output_prob
        #self.mask_head = copy.deepcopy(conv_head[0]) # discard pooling layer from conv_head
        self.mask_head = conv_head[0] # weights are shared with conv_head, pooling layer is discarded
        self.roi_align = roi_align
        self.transposed_conv = torch.nn.ConvTranspose2d(2048,256,2,stride=2,padding=0)
        self.classif_logits = torch.nn.Conv2d(256,81,1,stride=1,padding=0)
        self.relu = torch.nn.ReLU(inplace=True)
        self.sigmoid=torch.nn.Sigmoid()

    def forward(self, x, rois):
        x = self.roi_align(x,rois)
        x = self.mask_head(x)
        x = self.relu(self.transposed_conv(x))
        x = self.classif_logits(x)
        if self.output_prob:
            x = self.sigmoid(x)
        return x

class rpn_head(torch.nn.Module):
    def __init__(self):
        super(rpn_head, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.sigmoid = torch.nn.Sigmoid()
        self.conv_rpn = torch.nn.Conv2d(1024,1024,3,stride=1,padding=1)
        self.rpn_cls_prob = torch.nn.Conv2d(1024,15,1,stride=1,padding=0)
        self.rpn_bbox_pred = torch.nn.Conv2d(1024,60,1,stride=1,padding=0)
        self.proposal_generator = GenerateProposals()

    def forward(self, x, h, w, scaling_factor):
        conv_rpn = self.relu(self.conv_rpn(x))
        rpn_cls_prob = self.sigmoid(self.rpn_cls_prob(conv_rpn))
        rpn_bbox_pred = self.rpn_bbox_pred(conv_rpn)
        rpn_rois, rpn_roi_probs = self.proposal_generator(rpn_cls_prob,rpn_bbox_pred,h,w,scaling_factor)
        return rpn_rois

class detector(torch.nn.Module):
    def __init__(self,
                 arch='resnet50',
                 conv_body=['conv1','bn1','relu','maxpool','layer1','layer2','layer3'],
                 conv_head=['layer4','avgpool'],
                 use_rpn_head = False,
                 use_mask_head = False,
                 roi_feature_channels = 2048,
                 N_classes = 81,
                 caffe_pkl_file = None,
                 mapping_file = None,
                 output_prob = True):
        super(detector, self).__init__() 
        # Flags
        self.use_rpn_head = use_rpn_head
        self.use_mask_head = use_mask_head
        self.output_prob=output_prob

        ## Create main conv model
        if arch.startswith('resnet'):
            self.model = eval('models.'+arch+'()') # construct ResNet model (maybe not very safe :) 

            # swap stride (2,2) and (1,1) in first layers (PyTorch ResNet is slightly different to caffe2 ResNet)
            # this is required for compatibility with caffe2 models
            self.model.layer2[0].conv1.stride=(2,2)
            self.model.layer2[0].conv2.stride=(1,1)
            self.model.layer3[0].conv1.stride=(2,2)
            self.model.layer3[0].conv2.stride=(1,1)
            self.model.layer4[0].conv1.stride=(2,2)
            self.model.layer4[0].conv2.stride=(1,1)
        else:
            raise('Only resnet implemented so far!')
        # divide model into conv_body and conv_head
        self.conv_body=torch.nn.Sequential(*[getattr(self.model,l) for l in conv_body])
        self.conv_head=torch.nn.Sequential(*[getattr(self.model,l) for l in conv_head])        

        ## Create heads
        # RPN head
        if self.use_rpn_head:
            self.rpn = rpn_head()        
        # BBOX head
        self.bbox_head=torch.nn.Linear(roi_feature_channels,4*N_classes)
        # CLS head
        self.classif_head=torch.nn.Linear(roi_feature_channels,N_classes)                    
        # ROI cropping layer
        self.roi_align = RoIAlign(pooled_height=14,pooled_width=14,spatial_scale=0.0625)
        # MASK head
        if self.use_mask_head:
            self.mask_head = mask_head(self.conv_head,self.roi_align,output_prob)
        # load pretrained weights
        if caffe_pkl_file is not None:
            self.load_pretrained_weights(caffe_pkl_file, mapping_file)        

    
        self.model.eval() # this is needed as batch norm layers in caffe are only affine layers
        
    def forward(self, image, rois=None, scaling_factor=None):
        # store image size
        h,w = image.size(2), image.size(3)
        # compute dense conv features
        img_features = self.conv_body(image) # equivalent to gpu_0/res4_5_sum
        # generate rois if equipped with RPN head
        if self.use_rpn_head:
            rois = self.rpn(img_features,h,w,scaling_factor)
        # compute dense roi features
        roi_features = self.roi_align(img_features,rois) # 14x14 feature per proposal
        # compute 1x1 roi features
        roi_features = self.conv_head(roi_features) # 1x1 feature per proposal
        roi_features = roi_features.view(roi_features.size(0),-1)
        # compute classification scores
        class_scores = self.classif_head(roi_features)
        # compute classification probabilities
        if self.output_prob:
            class_scores = torch.nn.Softmax(dim=1)(class_scores)
        # compute bounding box parameters 
        bbox_deltas = self.bbox_head(roi_features)
        
        return (class_scores,bbox_deltas,rois,img_features)

    def load_pretrained_weights(self, caffe_pkl_file, mapping_file):
        ## Load pretrained weights
        
        # load caffe weights
        with open(caffe_pkl_file, 'rb') as f:
            caffe_data = pickle.load(f,encoding='latin1')
        # load pytorch<->caffe2 layer mapping
        mapping = np.load(mapping_file)
        # copy weights
        model_dict = self.model.state_dict()
        for i in range(len(mapping)):
            pytorch_key = mapping[i][0]
            caffe2_key = mapping[i][1]
            if model_dict[pytorch_key].size()==torch.FloatTensor(caffe_data['blobs'][caffe2_key]).size():
                if i==0: # convert from BGR to RGB
                    model_dict[pytorch_key]=torch.FloatTensor(caffe_data['blobs'][caffe2_key][:,(2, 1, 0),:,:])
                    #model_dict[pytorch_key]=torch.FloatTensor(caffe_data['blobs'][caffe2_key])
                else:
                    model_dict[pytorch_key]=torch.FloatTensor(caffe_data['blobs'][caffe2_key])
            else:
                print(str(i)+','+pytorch_key+','+caffe2_key)
                raise('size mistmatch')
        # update model
        self.model.load_state_dict(model_dict)
        # load also output head weights
        self.bbox_head.weight.data=torch.FloatTensor(caffe_data['blobs']['bbox_pred_w'])
        self.bbox_head.bias.data=torch.FloatTensor(caffe_data['blobs']['bbox_pred_b'])
        self.classif_head.weight.data=torch.FloatTensor(caffe_data['blobs']['cls_score_w'])
        self.classif_head.bias.data=torch.FloatTensor(caffe_data['blobs']['cls_score_b'])
        # load RPN weights
        if self.use_rpn_head:
            self.rpn.conv_rpn.weight.data = torch.FloatTensor(caffe_data['blobs']['conv_rpn_w'])
            self.rpn.conv_rpn.bias.data = torch.FloatTensor(caffe_data['blobs']['conv_rpn_b'])
            self.rpn.rpn_cls_prob.weight.data = torch.FloatTensor(caffe_data['blobs']['rpn_cls_logits_w'])
            self.rpn.rpn_cls_prob.bias.data = torch.FloatTensor(caffe_data['blobs']['rpn_cls_logits_b'])
            self.rpn.rpn_bbox_pred.weight.data = torch.FloatTensor(caffe_data['blobs']['rpn_bbox_pred_w'])
            self.rpn.rpn_bbox_pred.bias.data = torch.FloatTensor(caffe_data['blobs']['rpn_bbox_pred_b'])
        if self.use_mask_head:
            self.mask_head.transposed_conv.weight.data = torch.FloatTensor(caffe_data['blobs']['conv5_mask_w'])
            self.mask_head.transposed_conv.bias.data = torch.FloatTensor(caffe_data['blobs']['conv5_mask_b'])
            self.mask_head.classif_logits.weight.data = torch.FloatTensor(caffe_data['blobs']['mask_fcn_logits_w'])
            self.mask_head.classif_logits.bias.data = torch.FloatTensor(caffe_data['blobs']['mask_fcn_logits_b'])            


