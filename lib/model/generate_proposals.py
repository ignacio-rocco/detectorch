import torch
from torch.autograd import Variable
import numpy as np
import utils.boxes as box_utils
from utils.generate_anchors import generate_anchors


class GenerateProposals(torch.nn.Module):
    """Output object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """
    
    def __init__(self, spatial_scale=0.0625,
                 train=False,
                 rpn_pre_nms_top_n = None,
                 rpn_post_nms_top_n = None,
                 rpn_nms_thresh = None,
                 rpn_min_size = 0,
                 anchor_sizes=(32, 64, 128, 256, 512), 
                 anchor_aspect_ratios=(0.5, 1, 2)):
        super(GenerateProposals, self).__init__()
        self._anchors = generate_anchors(sizes=anchor_sizes, aspect_ratios=anchor_aspect_ratios,stride=1. / spatial_scale)
        self._num_anchors = self._anchors.shape[0]
        self._spatial_scale = spatial_scale
        self._train = train        
        self.rpn_pre_nms_top_n = rpn_pre_nms_top_n if rpn_pre_nms_top_n is not None else (12000 if train else 6000)
        self.rpn_post_nms_top_n = rpn_post_nms_top_n if rpn_post_nms_top_n is not None else (2000 if train else 1000)
        self.rpn_nms_thresh = rpn_nms_thresh if rpn_nms_thresh is not None else 0.7
        self.rpn_min_size = rpn_min_size if rpn_min_size is not None else 0

    def forward(self, rpn_cls_probs, rpn_bbox_pred, im_height, im_width, scaling_factor, spatial_scale=None):
        if spatial_scale is None:  
            spatial_scale = self._spatial_scale
        """See modeling.detector.GenerateProposals for inputs/outputs
        documentation.
        """
        # 1. for each location i in a (H, W) grid:
        #      generate A anchor boxes centered on cell i
        #      apply predicted bbox deltas to each of the A anchors at cell i
        # 2. clip predicted boxes to image
        # 3. remove predicted boxes with either height or width < threshold
        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take the top pre_nms_topN proposals before NMS
        # 6. apply NMS with a loose threshold (0.7) to the remaining proposals
        # 7. take after_nms_topN proposals after NMS
        # 8. return the top proposals
        
        # 1. get anchors at all features positions
        all_anchors_np = self.get_all_anchors(num_images = rpn_cls_probs.shape[0],
                                      feature_height = rpn_cls_probs.shape[2],
                                      feature_width = rpn_cls_probs.shape[3],
                                      spatial_scale = spatial_scale)
        
        all_anchors = Variable(torch.FloatTensor(all_anchors_np))
        if rpn_cls_probs.is_cuda:
            all_anchors = all_anchors.cuda()
    
        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        #   - bbox deltas will be (4 * A, H, W) format from conv output
        #   - transpose to (H, W, 4 * A)
        #   - reshape to (H * W * A, 4) where rows are ordered by (H, W, A)
        #     in slowest to fastest order to match the enumerated anchors
        bbox_deltas = rpn_bbox_pred.squeeze(0).permute(1, 2, 0).contiguous().view(-1, 4)
        bbox_deltas_np = bbox_deltas.cpu().data.numpy()

        # Same story for the scores:
        #   - scores are (A, H, W) format from conv output
        #   - transpose to (H, W, A)
        #   - reshape to (H * W * A, 1) where rows are ordered by (H, W, A)
        #     to match the order of anchors and bbox_deltas
        scores = rpn_cls_probs.squeeze(0).permute(1, 2, 0).contiguous().view(-1, 1)
        scores_np = scores.cpu().data.numpy()

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        if self.rpn_pre_nms_top_n <= 0 or self.rpn_pre_nms_top_n >= len(scores_np):
            order = np.argsort(-scores_np.squeeze())
        else:
            # Avoid sorting possibly large arrays; First partition to get top K
            # unsorted and then sort just those (~20x faster for 200k scores)
            inds = np.argpartition(
                -scores_np.squeeze(), self.rpn_pre_nms_top_n
            )[:self.rpn_pre_nms_top_n]
            order = np.argsort(-scores_np[inds].squeeze())
            order = inds[order]
            
        bbox_deltas = bbox_deltas[order, :]
        bbox_deltas_np = bbox_deltas_np[order, :]
        scores = scores[order,:]        
        scores_np = scores_np[order,:]
        all_anchors = all_anchors[order, :]
        all_anchors_np00 = all_anchors_np[order, :]    

        # Transform anchors into proposals via bbox transformations
        proposals = self.bbox_transform(all_anchors, bbox_deltas, (1.0, 1.0, 1.0, 1.0))

        # 2. clip proposals to image (may result in proposals with zero area
        # that will be removed in the next step)
        proposals = self.clip_tiled_boxes(proposals, im_height, im_width)
        proposals_np = proposals.cpu().data.numpy()

        # 3. remove predicted boxes with either height or width < min_size
        keep = self.filter_boxes(proposals_np, self.rpn_min_size, scaling_factor, im_height, im_width)

        proposals = proposals[keep, :]
        proposals_np = proposals_np[keep, :]
        scores = scores[keep,:]
        scores_np = scores_np[keep]

        # 6. apply loose nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        if self.rpn_nms_thresh > 0:
            keep = box_utils.nms(np.hstack((proposals_np, scores_np)), self.rpn_nms_thresh)
            if self.rpn_post_nms_top_n > 0:
                keep = keep[:self.rpn_post_nms_top_n]
                
            proposals = proposals[keep, :]
            scores = scores[keep,:]
            
        return proposals, scores
        
    def get_all_anchors(self,num_images,feature_height,feature_width,spatial_scale):
        # 1. Generate proposals from bbox deltas and shifted anchors
        # the number of proposals is equal to the number of anchors (eg.15)
        # times the feature support size (eg=50x50=2500), totaling about 40k

        # Enumerate all shifted positions on the (H, W) grid
        feat_stride = 1. / spatial_scale
        shift_x = np.arange(0, feature_width) * feat_stride
        shift_y = np.arange(0, feature_height) * feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y, copy=False)
        # Convert to (K, 4), K=H*W, where the columns are (dx, dy, dx, dy)
        # shift pointing to each grid location
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()

        # Broacast anchors over shifts to enumerate all anchors at all positions
        # in the (H, W) grid:
        #   - add A anchors of shape (1, A, 4) to
        #   - K shifts of shape (K, 1, 4) to get
        #   - all shifted anchors of shape (K, A, 4)
        #   - reshape to (K*A, 4) shifted anchors
        A = self._num_anchors
        K = shifts.shape[0]
        all_anchors = self._anchors[np.newaxis, :, :] + shifts[:, np.newaxis, :]
        all_anchors = all_anchors.reshape((K * A, 4))
        return all_anchors
        
    def filter_boxes(self, boxes, min_size, scale_factor, image_height, image_width):
        """Only keep boxes with both sides >= min_size and center within the image.
        """
        # Scale min_size to match image scale
        min_size *= scale_factor
        ws = boxes[:, 2] - boxes[:, 0] + 1
        hs = boxes[:, 3] - boxes[:, 1] + 1
        x_ctr = boxes[:, 0] + ws / 2.
        y_ctr = boxes[:, 1] + hs / 2.
        keep = np.where(
            (ws >= min_size) & (hs >= min_size) &
            (x_ctr < image_width) & (y_ctr < image_height))[0]
        return keep

    def bbox_transform(self, boxes, deltas, weights=(1.0, 1.0, 1.0, 1.0), clip_value=4.135166556742356):
        """Forward transform that maps proposal boxes to predicted ground-truth
        boxes using bounding-box regression deltas. See bbox_transform_inv for a
        description of the weights argument.
        """
        if boxes.size(0) == 0:
            return None
            #return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

        # get boxes dimensions and centers
        widths = boxes[:, 2] - boxes[:, 0] + 1.0
        heights = boxes[:, 3] - boxes[:, 1] + 1.0
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh
        
        clip_value = Variable(torch.FloatTensor([clip_value]))
        if boxes.is_cuda:
            clip_value = clip_value.cuda()

        # Prevent sending too large values into np.exp()
        dw = torch.min(dw,clip_value)
        dh = torch.min(dh,clip_value)

        pred_ctr_x = dx * widths.unsqueeze(1) + ctr_x.unsqueeze(1)
        pred_ctr_y = dy * heights.unsqueeze(1) + ctr_y.unsqueeze(1)
        pred_w = torch.exp(dw) * widths.unsqueeze(1)
        pred_h = torch.exp(dh) * heights.unsqueeze(1)

        # pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
        # x1
        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        # y1
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        # x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w - 1
        # y2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h - 1

        pred_boxes = torch.cat((pred_boxes_x1,
                                pred_boxes_y1,
                                pred_boxes_x2,
                                pred_boxes_y2),1)

        return pred_boxes

    def clip_tiled_boxes(self, boxes, image_height, image_width):
        """Clip boxes to image boundaries. im_shape is [height, width] and boxes
        has shape (N, 4 * num_tiled_boxes)."""

        im_w = Variable(torch.FloatTensor([float(image_width)]))
        im_h = Variable(torch.FloatTensor([float(image_height)]))
        z = Variable(torch.FloatTensor([0]))

        if boxes.is_cuda:
            im_w = im_w.cuda()
            im_h = im_h.cuda()
            z = z.cuda()
            
        # x1 >= 0
        boxes[:, 0::4] = torch.max(torch.min(boxes[:, 0::4], im_w - 1), z)
        # y1 >= 0
        boxes[:, 1::4] = torch.max(torch.min(boxes[:, 1::4], im_h - 1), z)
        # x2 < im_shape[1]
        boxes[:, 2::4] = torch.max(torch.min(boxes[:, 2::4], im_w - 1), z)
        # y2 < im_shape[0]
        boxes[:, 3::4] = torch.max(torch.min(boxes[:, 3::4], im_h - 1), z)
        
        return boxes