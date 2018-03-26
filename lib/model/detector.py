import torch
import pickle
import numpy as np
import torchvision.models as models
from model.roi_align import RoIAlignFunction, preprocess_rois
from model.generate_proposals import GenerateProposals
from model.collect_and_distribute_fpn_rpn_proposals import CollectAndDistributeFpnRpnProposals
from utils.utils import parse_th_to_caffe2
from torch.autograd import Variable
from math import log2

class fpn_body(torch.nn.Module):
    def __init__(self, conv_body, conv_body_layers, fpn_layers):
        super(fpn_body, self).__init__()
        self.conv_body = conv_body
        # Lateral convolution layers. This is not run as a sequential model. Sequential is just used to group the layers together.
        self.fpn_lateral = torch.nn.Sequential(*[torch.nn.Conv2d(in_channels=conv_body[conv_body_layers.index(l)][-1].bn3.num_features,
                                                                 out_channels=256,
                                                                 kernel_size=1,
                                                                 stride=1,
                                                                 padding=0) for l in fpn_layers])
        # make output convolutions. This is not run as a sequential model. Sequential is just used to group the layers together.
        self.fpn_output = torch.nn.Sequential(*[torch.nn.Conv2d(in_channels=256,
                                                                 out_channels=256,
                                                                 kernel_size=3,
                                                                 stride=1,
                                                                 padding=1) for l in fpn_layers])
        # upsampling layer
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        # store fpn layer indices 
        self.fpn_indices = [conv_body_layers.index(l) for l in fpn_layers]
        # keep fpn layer names
        self.fpn_layers = fpn_layers

    def forward(self, x):
        lateral=[]
        # do forward pass on the whole conv body, and store tensors for lateral computation
        for i in range(len(self.conv_body)):
            x=self.conv_body[i](x)
            if i in self.fpn_indices:
                lateral.append(x)
        # do lateral convolutions
        for i in range(len(self.fpn_lateral)):
            lateral[i]=self.fpn_lateral[i](lateral[i])
        # do top-down pass
        for i in range(len(self.fpn_lateral)-2, -1, -1):
            lateral[i]=self.upsample(lateral[i+1])+lateral[i]
        # do output convolutions
        for i in range(len(self.fpn_lateral)):
            lateral[i]=self.fpn_output[i](lateral[i])

        return lateral

class two_layer_mlp_head(torch.nn.Module):
    def __init__(self):
        super(two_layer_mlp_head, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.fc6 = torch.nn.Linear(256*7*7,1024)
        self.fc7 = torch.nn.Linear(1024,1024)

    def forward(self, x):
        x = x.view(x.size(0),-1) 
        x = self.relu(self.fc6(x))
        x = self.relu(self.fc7(x))
        return x

class four_layer_conv(torch.nn.Module):
    def __init__(self):
        super(four_layer_conv, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.fcn1 = torch.nn.Conv2d(256,256,3,stride=1,padding=1)
        self.fcn2 = torch.nn.Conv2d(256,256,3,stride=1,padding=1)
        self.fcn3 = torch.nn.Conv2d(256,256,3,stride=1,padding=1)
        self.fcn4 = torch.nn.Conv2d(256,256,3,stride=1,padding=1)
      

    def forward(self, x):
        x = self.relu(self.fcn1(x))
        x = self.relu(self.fcn2(x))
        x = self.relu(self.fcn3(x))
        x = self.relu(self.fcn4(x))
        return x

class mask_head(torch.nn.Module):
    def __init__(self, conv_head, roi_spatial_scale, roi_sampling_ratio, output_prob):
        super(mask_head, self).__init__()
        self.output_prob = output_prob
        self.conv_head = conv_head
        self.transposed_conv = torch.nn.ConvTranspose2d(256 if isinstance(conv_head,four_layer_conv) else 2048,256,2,stride=2,padding=0)
        self.classif_logits = torch.nn.Conv2d(256,81,1,stride=1,padding=0)
        self.relu = torch.nn.ReLU(inplace=True)
        self.sigmoid=torch.nn.Sigmoid()
        self.use_fpn = isinstance(roi_spatial_scale,list)
        self.roi_spatial_scale = roi_spatial_scale
        self.roi_sampling_ratio = roi_sampling_ratio
        self.roi_height = 14
        self.roi_width = 14

    def forward(self, x, rois, roi_original_idx=None):
        if self.use_fpn==False:
            x = RoIAlignFunction.apply(x, preprocess_rois(rois), self.roi_height, self.roi_width, self.roi_spatial_scale, self.roi_sampling_ratio) # 14x14 feature per proposal
        else:
            x = [RoIAlignFunction.apply(x[i], preprocess_rois(rois[i]), self.roi_height, self.roi_width, 
                                        self.roi_spatial_scale[i], self.roi_sampling_ratio) if rois[i] is not None else None for i in range(len(rois))]
            x = torch.cat(tuple(filter(lambda z: z is not None, x)),0)
            x = x[roi_original_idx,:]
        x = self.conv_head(x)
        x = self.relu(self.transposed_conv(x))
        x = self.classif_logits(x)
        if self.output_prob:
            x = self.sigmoid(x)
        return x

class rpn_head(torch.nn.Module):
    def __init__(self,in_channels=1024,out_channels=1024,n_anchors=15):
        super(rpn_head, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.sigmoid = torch.nn.Sigmoid()
        self.conv_rpn = torch.nn.Conv2d(in_channels,out_channels,3,stride=1,padding=1)
        self.rpn_cls_prob = torch.nn.Conv2d(out_channels,n_anchors,1,stride=1,padding=0)
        self.rpn_bbox_pred = torch.nn.Conv2d(out_channels,4*n_anchors,1,stride=1,padding=0)

    def forward(self, x):
        conv_rpn = self.relu(self.conv_rpn(x))
        rpn_cls_prob = self.sigmoid(self.rpn_cls_prob(conv_rpn))
        rpn_bbox_pred = self.rpn_bbox_pred(conv_rpn)
        return rpn_cls_prob,rpn_bbox_pred

class detector(torch.nn.Module):
    def __init__(self,
                 train=False,
                 arch='resnet50',
                 conv_body_layers=['conv1','bn1','relu','maxpool','layer1','layer2','layer3'],
                 # conv head can be a list of modules to use from the main model 
                 # or can be the string 'two_layer_mlp'
                 conv_head_layers=['layer4','avgpool'], 
                 # fpn layers is a list of the layers of the conv body used to define the levels of the FPN
                 fpn_layers = [],
                 fpn_extra_lvl = True, # add additional fpn lvl by 2x subsampling the last level
                 use_rpn_head = False,
                 use_mask_head = False,
                 mask_head_type = 'upshare',
                 roi_feature_channels = 2048,
                 N_classes = 81,
                 detector_pkl_file = None,
                 base_cnn_pkl_file = None,
                 output_prob = True,
                 roi_height = 14,
                 roi_width = 14,
                 roi_spatial_scale = 0.0625,
                 roi_sampling_ratio = 0):
        super(detector, self).__init__() 
        # RoI Parameters
        self.roi_height = int(roi_height)
        self.roi_width = int(roi_width)
        self.roi_spatial_scale = [float(i) for i in roi_spatial_scale] if isinstance(roi_spatial_scale, list) else float(roi_spatial_scale) 
        self.roi_sampling_ratio = int(roi_sampling_ratio)
        # Flags        
        self.train=train
        self.mask_head_type = mask_head_type
        self.use_fpn_body = len(fpn_layers)>0
        self.fpn_extra_lvl = fpn_extra_lvl
        self.use_rpn_head = use_rpn_head
        self.use_mask_head = use_mask_head
        self.use_two_layer_mlp_head = conv_head_layers=='two_layer_mlp'
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
        self.conv_body=torch.nn.Sequential(*[getattr(self.model,l) for l in conv_body_layers])
        # wrap in FPN model if needed
        if self.use_fpn_body:
            self.conv_body = fpn_body(self.conv_body, conv_body_layers, fpn_layers)
        if conv_head_layers=='two_layer_mlp':
            self.conv_head=two_layer_mlp_head()
        else:
            self.conv_head=torch.nn.Sequential(*[getattr(self.model,l) for l in conv_head_layers])        

        ## Create heads
        # RPN head
        if self.use_rpn_head and not self.use_fpn_body:
            self.rpn = rpn_head()
            self.proposal_generator = GenerateProposals(train=self.train)
        if self.use_rpn_head and self.use_fpn_body:
            self.rpn = rpn_head(in_channels=256,out_channels=256,n_anchors=3)
            spatial_scales = self.roi_spatial_scale
            if self.fpn_extra_lvl:
                spatial_scales = spatial_scales + [spatial_scales[-1]/2.]
            self.proposal_generator = [GenerateProposals(train=self.train,
                                                         spatial_scale=spatial_scales[i],
                                                         anchor_sizes=(32*2**i,),
                                                         rpn_pre_nms_top_n=12000 if self.train else 1000,
                                                         rpn_post_nms_top_n=2000 if self.train else 1000) for i in range(len(spatial_scales))]
            # Note, even when using the extra fpn level, proposals are note collected at this level
            self.collect_and_distr_rois = CollectAndDistributeFpnRpnProposals(spatial_scales=self.roi_spatial_scale,train=self.train)

        # bounding box regression head
        self.bbox_head=torch.nn.Linear(roi_feature_channels,4*N_classes)
        # classification head
        self.classif_head=torch.nn.Linear(roi_feature_channels,N_classes)                    
        # mask prediction head
        if self.use_mask_head:
            if self.mask_head_type == 'upshare':
                mask_head_conv = self.conv_head[0]
            elif self.mask_head_type == '1up4convs':
                mask_head_conv = four_layer_conv()
            self.mask_head = mask_head(mask_head_conv,
                                       self.roi_spatial_scale,
                                       self.roi_sampling_ratio, output_prob)
        # load pretrained weights
        if detector_pkl_file is not None:
            self.load_pretrained_weights(detector_pkl_file, model='detector')
        elif base_cnn_pkl_file is not None:
            self.load_pretrained_weights(base_cnn_pkl_file, model='base_cnn')

    
        self.model.eval() # this is needed as batch norm layers in caffe are only affine layers (no running mean or std)
        
    def forward(self, image, rois=None, scaling_factor=None, roi_original_idx=None):
        h,w = image.size(2), image.size(3)

        # compute dense conv features
        img_features = self.conv_body(image) # equivalent to gpu_0/res4_5_sum

        # generate rois if equipped with RPN head
        if self.use_rpn_head and not self.use_fpn_body:
            # case without FPN, proposal generation in a single step
            rpn_cls_prob,rpn_bbox_pred = self.rpn(img_features)
            rois, rpn_roi_probs = self.proposal_generator(rpn_cls_prob,rpn_bbox_pred,h,w,scaling_factor)
        elif self.use_rpn_head and self.use_fpn_body:
            # case with FPN, proposal generation for each FPN level
            assert isinstance(img_features,list) and isinstance(self.roi_spatial_scale,list)
            img_features_tmp = img_features
            if self.fpn_extra_lvl:
                # add extra feature resolution by subsampling the last FPN feature level
                img_features_tmp = img_features_tmp + [torch.nn.functional.max_pool2d(img_features[-1],1,stride=2)]
            cls_and_bbox = [self.rpn(img_features_tmp[i]) for i in range(len(img_features_tmp))]
            rois_and_probs = [self.proposal_generator[i](cls_and_bbox[i][0],cls_and_bbox[i][1],h,w,scaling_factor) for i in range(len(img_features_tmp))]
            rois = [item[0] for item in rois_and_probs]
            rpn_roi_probs = [item[1] for item in rois_and_probs]
            # we now combine rois from all FPN levels and re-assign to correct FPN level for later RoI pooling
            rois,roi_original_idx = self.collect_and_distr_rois(rois,rpn_roi_probs)

        # compute dense roi features
        if self.use_fpn_body==False:        
            roi_features = RoIAlignFunction.apply(img_features, preprocess_rois(rois), self.roi_height, self.roi_width, self.roi_spatial_scale, self.roi_sampling_ratio) # 14x14 feature per proposal
        else:
            assert isinstance(img_features,list) and isinstance(rois,list) and isinstance(self.roi_spatial_scale,list)
            roi_features = [RoIAlignFunction.apply(img_features[i], preprocess_rois(rois[i]),self.roi_height, self.roi_width, 
                                             self.roi_spatial_scale[i], self.roi_sampling_ratio) for i in range(len(self.roi_spatial_scale))]
            # concatenate roi features from all levels of FPN
            roi_features = torch.cat(tuple(roi_features),0)
            rois = torch.cat(tuple(rois),0)
            # restore original order
            roi_features = roi_features[roi_original_idx,:]
            rois = rois[roi_original_idx,:]

        # compute 1x1 roi features
        roi_features = self.conv_head(roi_features) # 1x1 feature per proposal
        roi_features = roi_features.view(roi_features.size(0),-1)

        # compute classification scores
        cls_score = self.classif_head(roi_features)

        # compute classification probabilities
        if self.output_prob:
            cls_score = torch.nn.functional.softmax(cls_score,dim=1)

        # compute bounding box parameters 
        bbox_pred = self.bbox_head(roi_features)
        
        return (cls_score,bbox_pred,rois,img_features)

    
    def load_pretrained_weights(self, caffe_pkl_file, model='detector'):
        ## Load pretrained weights
        print('Loading pretrained weights:')
        # load caffe weights
        with open(caffe_pkl_file, 'rb') as f:
            caffe_data = pickle.load(f,encoding='latin1')
        if model=='detector':
            caffe_data=caffe_data['blobs']

        model_dict = self.model.state_dict()
        print('-> loading conv. body weights')
        for k in model_dict.keys():
            if 'running' in k or 'fc' in k: # skip running mean/std and fc weights
                continue
            k_caffe = parse_th_to_caffe2(k.split('.'))
            assert model_dict[k].size()==torch.FloatTensor(caffe_data[k_caffe]).size()
            if k=='conv1.weight': # convert from BGR to RGB                
                model_dict[k]=torch.FloatTensor(caffe_data[k_caffe][:,(2, 1, 0),:,:])
            else:
                model_dict[k]=torch.FloatTensor(caffe_data[k_caffe])
        # update model
        self.model.load_state_dict(model_dict)
        # only if full detector model was loaded, as opposed to loading an ImageNet pretrained base-cnn only
        if model=='detector':
            print('-> loading output head weights')
            # load also output head weights
            self.bbox_head.weight.data=torch.FloatTensor(caffe_data['bbox_pred_w'])
            self.bbox_head.bias.data=torch.FloatTensor(caffe_data['bbox_pred_b'])
            self.classif_head.weight.data=torch.FloatTensor(caffe_data['cls_score_w'])
            self.classif_head.bias.data=torch.FloatTensor(caffe_data['cls_score_b'])
            # load RPN weights
            if self.use_rpn_head and not self.use_fpn_body:
                print('-> loading rpn head weights')
                self.rpn.conv_rpn.weight.data = torch.FloatTensor(caffe_data['conv_rpn_w'])
                self.rpn.conv_rpn.bias.data = torch.FloatTensor(caffe_data['conv_rpn_b'])
                self.rpn.rpn_cls_prob.weight.data = torch.FloatTensor(caffe_data['rpn_cls_logits_w'])
                self.rpn.rpn_cls_prob.bias.data = torch.FloatTensor(caffe_data['rpn_cls_logits_b'])
                self.rpn.rpn_bbox_pred.weight.data = torch.FloatTensor(caffe_data['rpn_bbox_pred_w'])
                self.rpn.rpn_bbox_pred.bias.data = torch.FloatTensor(caffe_data['rpn_bbox_pred_b'])
            if self.use_rpn_head and self.use_fpn_body:
                print('-> loading rpn head weights')
                self.rpn.conv_rpn.weight.data = torch.FloatTensor(caffe_data['conv_rpn_fpn2_w'])
                self.rpn.conv_rpn.bias.data = torch.FloatTensor(caffe_data['conv_rpn_fpn2_b'])
                self.rpn.rpn_cls_prob.weight.data = torch.FloatTensor(caffe_data['rpn_cls_logits_fpn2_w'])
                self.rpn.rpn_cls_prob.bias.data = torch.FloatTensor(caffe_data['rpn_cls_logits_fpn2_b'])
                self.rpn.rpn_bbox_pred.weight.data = torch.FloatTensor(caffe_data['rpn_bbox_pred_fpn2_w'])
                self.rpn.rpn_bbox_pred.bias.data = torch.FloatTensor(caffe_data['rpn_bbox_pred_fpn2_b'])
            if self.use_mask_head:
                print('-> loading mask head weights')                
                self.mask_head.transposed_conv.weight.data = torch.FloatTensor(caffe_data['conv5_mask_w'])
                self.mask_head.transposed_conv.bias.data = torch.FloatTensor(caffe_data['conv5_mask_b'])
                self.mask_head.classif_logits.weight.data = torch.FloatTensor(caffe_data['mask_fcn_logits_w'])
                self.mask_head.classif_logits.bias.data = torch.FloatTensor(caffe_data['mask_fcn_logits_b'])  
                if self.mask_head_type=='1up4convs':
                    print('-> loading 1up4convs mask head weights')                
                    self.mask_head.conv_head.fcn1.weight.data = torch.FloatTensor(caffe_data['_[mask]_fcn1_w'])
                    self.mask_head.conv_head.fcn1.bias.data   = torch.FloatTensor(caffe_data['_[mask]_fcn1_b'])
                    self.mask_head.conv_head.fcn2.weight.data = torch.FloatTensor(caffe_data['_[mask]_fcn2_w'])
                    self.mask_head.conv_head.fcn2.bias.data   = torch.FloatTensor(caffe_data['_[mask]_fcn2_b'])
                    self.mask_head.conv_head.fcn3.weight.data = torch.FloatTensor(caffe_data['_[mask]_fcn3_w'])
                    self.mask_head.conv_head.fcn3.bias.data   = torch.FloatTensor(caffe_data['_[mask]_fcn3_b'])
                    self.mask_head.conv_head.fcn4.weight.data = torch.FloatTensor(caffe_data['_[mask]_fcn4_w'])
                    self.mask_head.conv_head.fcn4.bias.data   = torch.FloatTensor(caffe_data['_[mask]_fcn4_b'])
            # load FPN weights
            if self.use_fpn_body:
                print('-> loading FPN lateral weights')
                for i in range(len(self.conv_body.fpn_layers)):
                    l=self.conv_body.fpn_layers[i]
                    # get name of last conv layer of each ResNet block which is used for FPN
                    k_caffe=parse_th_to_caffe2((l+'.'+list(getattr(self.model,l).state_dict().keys())[-1]).split('.'))
                    k_caffe=k_caffe[:k_caffe.rfind("_")]
                    if i<len(self.conv_body.fpn_layers)-1:
                        suffix='_sum_lateral'
                    else:
                        suffix='_sum'
                    self.conv_body.fpn_lateral[i].weight.data = torch.FloatTensor(caffe_data['fpn_inner_'+k_caffe+suffix+'_w'])
                    self.conv_body.fpn_lateral[i].bias.data = torch.FloatTensor(caffe_data['fpn_inner_'+k_caffe+suffix+'_b'])
                    self.conv_body.fpn_output[i].weight.data = torch.FloatTensor(caffe_data['fpn_'+k_caffe+'_sum_w'])
                    self.conv_body.fpn_output[i].bias.data = torch.FloatTensor(caffe_data['fpn_'+k_caffe+'_sum_b'])
            # load 2 layer mlp weights
            if self.use_two_layer_mlp_head:
                print('-> loading two layer mlp conv head...')
                self.conv_head.fc6.weight.data = torch.FloatTensor(caffe_data['fc6_w'])
                self.conv_head.fc6.bias.data = torch.FloatTensor(caffe_data['fc6_b'])
                self.conv_head.fc7.weight.data = torch.FloatTensor(caffe_data['fc7_w'])
                self.conv_head.fc7.bias.data = torch.FloatTensor(caffe_data['fc7_b'])



