import numpy as np
import torch
from utils.prep_im_for_blob import prep_im_for_blob


class preprocess_sample(object):
    # performs the preprocessing (including building image pyramids and scaling the coordinates)
    def __init__(self, target_sizes=800,max_size=1333,mean=[122.7717, 115.9465, 102.9801],remove_dup_proposals=True,spatial_scale=0.0625):
        self.mean=mean
        self.target_sizes=target_sizes if isinstance(target_sizes,list) else [target_sizes]
        self.max_size=max_size
        self.remove_dup_proposals=True
        self.spatial_scale=spatial_scale
        
    def __call__(self, sample):
        # resizes image and returns scale factors
        im_list,im_scales = prep_im_for_blob(sample['image'],
                                             pixel_means=self.mean,
                                             target_sizes=self.target_sizes,
                                             max_size=self.max_size)
        singlescale = len(self.target_sizes)==1
        sample['image'] = torch.FloatTensor(im_list[0]).permute(2,0,1) if singlescale else [[torch.FloatTensor(im).permute(2,0,1) for im in im_list]]
        sample['scaling_factors'] = torch.FloatTensor([im_scales[0]]) if singlescale else [[torch.FloatTensor([sc]) for sc in im_scales]]
        if  sample['gt_coords'] is not None:
            sample['gt_coords'] = sample['gt_coords']*im_scales[0] if singlescale else [[sample['gt_coords']*sc for sc in im_scales]]
        if len(sample['proposal_coords'])!=1:  # check that valid proposal coords have been given
            sample['proposal_coords'] = sample['proposal_coords']*im_scales[0] if singlescale else [[sample['proposal_coords']*sc for sc in im_scales]]
            if self.remove_dup_proposals:
                props_and_inv_index = self.remove_dup_prop(sample['proposal_coords']) if singlescale else [self.remove_dup_prop(prop) for prop in sample['proposal_coords'][0]]
                sample['proposal_coords'] = props_and_inv_index[0] if singlescale else [[p_and_idx[0] for p_and_idx in props_and_inv_index]]
                sample['proposal_inv_index'] = props_and_inv_index[1] if singlescale else [[p_and_idx[1] for p_and_idx in props_and_inv_index]]
        return sample

    # from Detectron test.py
    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    def remove_dup_prop(self,proposals): 
        proposals=proposals.numpy()
        v = np.array([1e3, 1e6, 1e9, 1e12])

        hashes = np.round(proposals * self.spatial_scale).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True, return_inverse=True)
        proposals = proposals[index, :]
        return (torch.FloatTensor(proposals),torch.FloatTensor(inv_index))