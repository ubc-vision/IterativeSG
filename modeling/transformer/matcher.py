import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import numpy as np

from .util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

from detectron2.utils.registry import Registry
from torchvision.ops.boxes import box_area

MATCHER_REGISTRY = Registry("MATCHER_REGISTRY")

@MATCHER_REGISTRY.register()
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets, return_cost_matrix=False):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()        

        sizes = [len(v["boxes"]) for v in targets]
        C_split = C.split(sizes, -1)
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C_split)]
        if not return_cost_matrix:
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        else:
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices], C_split

@MATCHER_REGISTRY.register()
class IterativeHungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets, return_cost_matrix=False, mask=None):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()        
        if mask is not None:
            C[:, ~mask] = np.float("inf")

        sizes = [len(v["boxes"]) for v in targets]
        C_split = C.split(sizes, -1)
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C_split)]
        
        if not return_cost_matrix:
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        else:
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices], C_split

    @torch.no_grad()
    def forward_relation(self, outputs, targets, return_cost_matrix=False):
        bs, num_queries = outputs["relation_logits"].shape[:2]
        out_prob = outputs["relation_logits"].flatten(0, 1).softmax(-1)
        out_sub_prob = outputs["relation_subject_logits"].flatten(0, 1).softmax(-1)
        out_obj_prob = outputs["relation_object_logits"].flatten(0, 1).softmax(-1)
        out_bbox = outputs["relation_boxes"].flatten(0, 1)
        out_sub_bbox = outputs["relation_subject_boxes"].flatten(0, 1)
        out_obj_bbox = outputs["relation_object_boxes"].flatten(0, 1)

        aux_out_prob = [output['pred_logits'].flatten(0, 1).softmax(-1) for output in outputs['aux_outputs_r']]
        aux_out_sub_prob = [output['pred_logits'].flatten(0, 1).softmax(-1) for output in outputs['aux_outputs_r_sub']]
        aux_out_obj_prob = [output['pred_logits'].flatten(0, 1).softmax(-1) for output in outputs['aux_outputs_r_obj']]
        aux_out_bbox = [output['pred_boxes'].flatten(0, 1) for output in outputs['aux_outputs_r']]
        aux_out_sub_bbox = [output['pred_boxes'].flatten(0, 1) for output in outputs['aux_outputs_r_sub']]
        aux_out_obj_bbox = [output['pred_boxes'].flatten(0, 1) for output in outputs['aux_outputs_r_obj']]

        device = out_prob.device

        gt_labels = [v['combined_labels'] for v in targets]
        gt_boxes = [v['combined_boxes'] for v in targets]
        relations = [v["image_relations"] for v in targets]
        relation_boxes = [v['relation_boxes'] for v in targets]
        
        if len(relations) > 0:
            tgt_ids = torch.cat(relations)[:, 2]
            tgt_sub_labels = torch.cat([gt_label[relation[:, 0]] for gt_label, relation in zip(gt_labels, relations)])
            tgt_obj_labels = torch.cat([gt_label[relation[:, 1]] for gt_label, relation in zip(gt_labels, relations)])
            tgt_boxes = torch.cat(relation_boxes)
            tgt_sub_boxes = torch.cat([gt_box[relation[:, 0]] for gt_box, relation in zip(gt_boxes, relations)])
            tgt_obj_boxes = torch.cat([gt_box[relation[:, 1]] for gt_box, relation in zip(gt_boxes, relations)])
        else:
            tgt_ids = torch.tensor([]).long().to(device)
            tgt_sub_labels = torch.tensor([]).long().to(device)
            tgt_obj_labels = torch.tensor([]).long().to(device)
            tgt_boxes = torch.zeros((0,4)).to(device)
            tgt_sub_boxes = torch.zeros((0,4)).to(device)
            tgt_obj_boxes = torch.zeros((0,4)).to(device)

        cost_class = -out_prob[:, tgt_ids]
        cost_subject_class = -out_sub_prob[:, tgt_sub_labels]
        cost_object_class = -out_obj_prob[:, tgt_obj_labels]
        
        cost_bbox = torch.cdist(out_bbox, tgt_boxes, p=1)
        cost_subject_bbox = torch.cdist(out_sub_bbox, tgt_sub_boxes, p=1)
        cost_object_bbox = torch.cdist(out_obj_bbox, tgt_obj_boxes, p=1)
        
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_boxes))
        cost_subject_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_sub_bbox), box_cxcywh_to_xyxy(tgt_sub_boxes))
        cost_object_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_obj_bbox), box_cxcywh_to_xyxy(tgt_obj_boxes))

        C = self.cost_bbox * (cost_bbox + cost_subject_bbox + cost_object_bbox) + self.cost_class * (cost_class + cost_subject_class + cost_object_class) + self.cost_giou * (cost_giou + cost_subject_giou + cost_object_giou)
        
        # Add aux loss cost 
        for aux_idx in range(len(aux_out_prob)):
            aux_cost_class = -aux_out_prob[aux_idx][:, tgt_ids]
            aux_cost_subject_class = -aux_out_sub_prob[aux_idx][:, tgt_sub_labels]
            aux_cost_object_class = -aux_out_obj_prob[aux_idx][:, tgt_obj_labels]

            aux_cost_bbox = torch.cdist(aux_out_bbox[aux_idx], tgt_boxes, p=1)
            aux_cost_subject_bbox = torch.cdist(aux_out_sub_bbox[aux_idx], tgt_sub_boxes, p=1)
            aux_cost_object_bbox = torch.cdist(aux_out_obj_bbox[aux_idx], tgt_obj_boxes, p=1)

            aux_cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(aux_out_bbox[aux_idx]), box_cxcywh_to_xyxy(tgt_boxes))
            aux_cost_subject_giou = -generalized_box_iou(box_cxcywh_to_xyxy(aux_out_sub_bbox[aux_idx]), box_cxcywh_to_xyxy(tgt_sub_boxes))
            aux_cost_object_giou = -generalized_box_iou(box_cxcywh_to_xyxy(aux_out_obj_bbox[aux_idx]), box_cxcywh_to_xyxy(tgt_obj_boxes))
            aux_C = self.cost_bbox * (aux_cost_bbox + aux_cost_subject_bbox + aux_cost_object_bbox) + self.cost_class * (aux_cost_class + aux_cost_subject_class + aux_cost_object_class) + self.cost_giou * (aux_cost_giou + aux_cost_subject_giou + aux_cost_object_giou)

            C = C + aux_C
            
        C = C.view(bs, num_queries, -1).cpu()   
        
        sizes = [len(v["image_relations"]) for v in targets]
        C_split = C.split(sizes, -1)
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C_split)]
        indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

        # Remaining GT objects matching
        pred_masks = {'subject': [], 'object': []}
        target_masks = {'subject' :[], 'object': []}
        combined_indices = {'subject' :[], 'object': [], 'relation': []}
        for image_idx, target in enumerate(targets):
            all_objects = torch.arange(len(gt_labels[image_idx])).to(device)
            relation = target['image_relations']
            curr_relation_idx = indices[image_idx]
            curr_pred_mask = torch.ones(num_queries, device=device)
            curr_pred_mask[curr_relation_idx[0]] = 0
            curr_pred_mask = (curr_pred_mask == 1)
            
            combined_indices['relation'].append((curr_relation_idx[0], curr_relation_idx[1]))
            for branch_idx, branch_type in enumerate(['subject', 'object']):  
                combined_indices[branch_type].append((curr_relation_idx[0], relation[:, branch_idx][curr_relation_idx[1]].cpu()))
        return combined_indices



def build_matcher(name, cost_class, cost_bbox, cost_giou, topk=1):
    if topk == 1:
        return MATCHER_REGISTRY.get(name)(cost_class=cost_class, cost_bbox=cost_bbox, cost_giou=cost_giou)
    else:
        return MATCHER_REGISTRY.get(name)(cost_class=cost_class, cost_bbox=cost_bbox, cost_giou=cost_giou, topk=topk)