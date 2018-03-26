import numpy as np
import torch
from utils.blob import prep_im_for_blob,im_list_to_blob
from utils.fast_rcnn_sample_rois import fast_rcnn_sample_rois
from utils.multilevel_rois import add_multilevel_rois_for_test

class preprocess_sample(object):
    # performs the preprocessing (including building image pyramids and scaling the coordinates)
    def __init__(self,
                 target_sizes=800,
                 max_size=1333,
                 mean=[122.7717, 115.9465, 102.9801],
                 remove_dup_proposals=True,
                 fpn_on=False,
                 spatial_scale=0.0625,
                 sample_proposals_for_training=False):
        self.mean=mean
        self.target_sizes=target_sizes if isinstance(target_sizes,list) else [target_sizes]
        self.max_size=max_size
        self.remove_dup_proposals=remove_dup_proposals
        self.fpn_on=fpn_on
        self.spatial_scale=spatial_scale
        self.sample_proposals_for_training = sample_proposals_for_training
        
    def __call__(self, sample):
        # resizes image and returns scale factors
        original_im_size=sample['image'].shape
        im_list,im_scales = prep_im_for_blob(sample['image'],
                                             pixel_means=self.mean,
                                             target_sizes=self.target_sizes,
                                             max_size=self.max_size)
        sample['image'] = torch.FloatTensor(im_list_to_blob(im_list,self.fpn_on)) # im_list_to blob swaps channels and adds stride in case of fpn
        sample['scaling_factors'] = im_scales[0] 
        sample['original_im_size'] = torch.FloatTensor(original_im_size)
        if len(sample['dbentry']['boxes'])!=0 and not self.sample_proposals_for_training: # Fast RCNN test
            proposals = sample['dbentry']['boxes']*im_scales[0]  
            if self.remove_dup_proposals:
                proposals,_ = self.remove_dup_prop(proposals) 
            
            if self.fpn_on==False:
                sample['rois'] = torch.FloatTensor(proposals)
            else:
                multiscale_proposals = add_multilevel_rois_for_test({'rois': proposals},'rois')
                for k in multiscale_proposals.keys():
                    sample[k] = torch.FloatTensor(multiscale_proposals[k])

        elif self.sample_proposals_for_training: # Fast RCNN training
            sampled_rois_labels_and_targets = fast_rcnn_sample_rois(roidb=sample['dbentry'],
                                                                    im_scale=im_scales[0],
                                                                    batch_idx=0) # ok as long as we keep batch_size=1
            sampled_rois_labels_and_targets = {key: torch.FloatTensor(value) for key,value in sampled_rois_labels_and_targets.items()}
            # add to sample
            sample = {**sample, **sampled_rois_labels_and_targets} 
        # remove dbentry from sample
        del sample['dbentry']
        return sample

    # from Detectron test.py
    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    def remove_dup_prop(self,proposals): 
        v = np.array([1e3, 1e6, 1e9, 1e12])

        hashes = np.round(proposals * self.spatial_scale).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True, return_inverse=True)
        proposals = proposals[index, :]

        return (proposals,inv_index)