# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

# some functions are from Detectron

import numpy as np
from torch.autograd import Variable
import utils.boxes as box_utils
import cv2
import pycocotools.mask as mask_util


def to_np(x):
    if isinstance(x,np.ndarray):
        return x    
    if isinstance(x,Variable):
        x=x.data
    return x.cpu().numpy()

def empty_results(num_classes, num_images):
    """Return empty results lists for boxes, masks, and keypoints.
    Box detections are collected into:
      all_boxes[cls][image] = N x 5 array with columns (x1, y1, x2, y2, score)
    Instance mask predictions are collected into:
      all_segms[cls][image] = [...] list of COCO RLE encoded masks that are in
      1:1 correspondence with the boxes in all_boxes[cls][image]
    Keypoint predictions are collected into:
      all_keyps[cls][image] = [...] list of keypoints results, each encoded as
      a 3D array (#rois, 4, #keypoints) with the 4 rows corresponding to
      [x, y, logit, prob] (See: utils.keypoints.heatmaps_to_keypoints).
      Keypoints are recorded for person (cls = 1); they are in 1:1
      correspondence with the boxes in all_boxes[cls][image].
    """
    # Note: do not be tempted to use [[] * N], which gives N references to the
    # *same* empty list.
    all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    all_segms = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    all_keyps = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    return all_boxes, all_segms, all_keyps


def extend_results(index, all_res, im_res):
    """Add results for an image to the set of all results at the specified
    index.
    """
    # Skip cls_idx 0 (__background__)
    for cls_idx in range(1, len(im_res)):
        all_res[cls_idx][index] = im_res[cls_idx]
        
# When mapping from image ROIs to feature map ROIs, there's some aliasing
# (some distinct image ROIs get mapped to the same feature ROI).
# Here, we identify duplicate feature ROIs, so we only compute features
# on the unique subset.
def remove_dup_prop(self,proposals): 
    proposals=proposals.data.numpy()
    v = np.array([1e3, 1e6, 1e9, 1e12])

    hashes = np.round(proposals * self.spatial_scale).dot(v)
    _, index, inv_index = np.unique(hashes, return_index=True, return_inverse=True)
    proposals = proposals[index, :]
    return torch.FloatTensor(proposals)


def postprocess_output(rois,scaling_factor,im_size,class_scores,bbox_deltas,bbox_reg_weights = (10.0,10.0,5.0,5.0)):
    boxes = to_np(rois.div(scaling_factor).squeeze(0))
    bbox_deltas = to_np(bbox_deltas)    
    orig_im_size = to_np(im_size).squeeze()    
    # apply deltas
    pred_boxes = box_utils.bbox_transform(boxes, bbox_deltas, bbox_reg_weights)
    # clip on boundaries
    pred_boxes = box_utils.clip_tiled_boxes(pred_boxes,orig_im_size)    
    scores = to_np(class_scores)
    # Map scores and predictions back to the original set of boxes
    # This re-duplicates the previously removed boxes
    # Is there any use for this?
#    inv_index = to_np(batch['proposal_inv_index']).squeeze().astype(np.int64)
#    scores = scores[inv_index, :]
#    pred_boxes = pred_boxes[inv_index, :]
    # threshold on score and run nms to remove duplicates
    scores_final, boxes_final, boxes_per_class = box_results_with_nms_and_limit(scores, pred_boxes)
    
    return (scores_final, boxes_final, boxes_per_class)

def box_results_with_nms_and_limit(scores, boxes,
                                   num_classes=81,
                                   score_thresh=0.05,
                                   overlap_thresh=0.5,
                                   do_soft_nms=False,
                                   soft_nms_sigma=0.5,
                                   soft_nms_method='linear',
                                   do_bbox_vote=False,
                                   bbox_vote_thresh=0.8,
                                   bbox_vote_method='ID',
                                   max_detections_per_img=100, ### over all classes ###
                                   ):
    """Returns bounding-box detection results by thresholding on scores and
    applying non-maximum suppression (NMS).
    
    A number of #detections presist after this and are returned, sorted by class

    `boxes` has shape (#detections, 4 * #classes), where each row represents
    a list of predicted bounding boxes for each of the object classes in the
    dataset (including the background class). The detections in each row
    originate from the same object proposal.

    `scores` has shape (#detection, #classes), where each row represents a list
    of object detection confidence scores for each of the object classes in the
    dataset (including the background class). `scores[i, j]`` corresponds to the
    box at `boxes[i, j * 4:(j + 1) * 4]`.
    """
    cls_boxes = [[] for _ in range(num_classes)]
    # Apply threshold on detection probabilities and apply NMS
    # Skip j = 0, because it's the background class
    for j in range(1, num_classes):
        inds = np.where(scores[:, j] > score_thresh)[0]
        scores_j = scores[inds, j]
        boxes_j = boxes[inds, j * 4:(j + 1) * 4]
        dets_j = np.hstack((boxes_j, scores_j[:, np.newaxis])).astype(
            np.float32, copy=False
        )
        if do_soft_nms:
            nms_dets, _ = box_utils.soft_nms(
                dets_j,
                sigma=soft_nms_sigma,
                overlap_thresh=overlap_thresh,
                score_thresh=0.0001,
                method=soft_nms_method
            )
        else:
            keep = box_utils.nms(dets_j, overlap_thresh)
            nms_dets = dets_j[keep, :]
        # Refine the post-NMS boxes using bounding-box voting
        if do_bbox_vote:
            nms_dets = box_utils.box_voting(
                nms_dets,
                dets_j,
                bbox_vote_thresh,
                scoring_method=bbox_vote_method
            )
        cls_boxes[j] = nms_dets

    # Limit to max_per_image detections **over all classes**
    if max_detections_per_img > 0:
        image_scores = np.hstack(
            [cls_boxes[j][:, -1] for j in range(1, num_classes)]
        )
        if len(image_scores) > max_detections_per_img:
            image_thresh = np.sort(image_scores)[-max_detections_per_img]
            for j in range(1, num_classes):
                keep = np.where(cls_boxes[j][:, -1] >= image_thresh)[0]
                cls_boxes[j] = cls_boxes[j][keep, :]

    im_results = np.vstack([cls_boxes[j] for j in range(1, num_classes)])
    boxes = im_results[:, :-1]
    scores = im_results[:, -1]
    return scores, boxes, cls_boxes

def segm_results(cls_boxes, masks, ref_boxes, im_h, im_w,
                 num_classes=81,
                 M=14, #  cfg.MRCNN.RESOLUTION
                 cls_specific_mask=True,
                 thresh_binarize=0.5):
    cls_segms = [[] for _ in range(num_classes)]
    mask_ind = 0
    # To work around an issue with cv2.resize (it seems to automatically pad
    # with repeated border values), we manually zero-pad the masks by 1 pixel
    # prior to resizing back to the original image resolution. This prevents
    # "top hat" artifacts. We therefore need to expand the reference boxes by an
    # appropriate factor.
    scale = (M + 2.0) / M
    ref_boxes = box_utils.expand_boxes(ref_boxes, scale)
    ref_boxes = ref_boxes.astype(np.int32)
    padded_mask = np.zeros((M + 2, M + 2), dtype=np.float32)

    # skip j = 0, because it's the background class
    for j in range(1, num_classes):
        segms = []
        for _ in range(cls_boxes[j].shape[0]):
            if cls_specific_mask:
                padded_mask[1:-1, 1:-1] = masks[mask_ind, j, :, :]
            else:
                padded_mask[1:-1, 1:-1] = masks[mask_ind, 0, :, :]

            ref_box = ref_boxes[mask_ind, :]
            w = ref_box[2] - ref_box[0] + 1
            h = ref_box[3] - ref_box[1] + 1
            w = np.maximum(w, 1)
            h = np.maximum(h, 1)

            mask = cv2.resize(padded_mask, (w, h))
            mask = np.array(mask > thresh_binarize, dtype=np.uint8)
            im_mask = np.zeros((im_h, im_w), dtype=np.uint8)

            x_0 = max(ref_box[0], 0)
            x_1 = min(ref_box[2] + 1, im_w)
            y_0 = max(ref_box[1], 0)
            y_1 = min(ref_box[3] + 1, im_h)

            im_mask[y_0:y_1, x_0:x_1] = mask[
                (y_0 - ref_box[1]):(y_1 - ref_box[1]),
                (x_0 - ref_box[0]):(x_1 - ref_box[0])
            ]

            # Get RLE encoding used by the COCO evaluation API
            rle = mask_util.encode(
                np.array(im_mask[:, :, np.newaxis], order='F')
            )[0]
            rle['counts'] = rle['counts'].decode() # convert back to str so that it can be later saved to json
            segms.append(rle)

            mask_ind += 1

        cls_segms[j] = segms

    assert mask_ind == masks.shape[0]
    return cls_segms