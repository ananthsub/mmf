# Copyright (c) Facebook, Inc. and its affiliates.

"""
 coding=utf-8
 Copyright 2018, Antonio Mendoza Hao Tan, Mohit Bansal
 Adapted From Facebook Inc, Detectron2 && Huggingface Co.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.import copy
 """

import copy
import itertools
import math
import os
import pickle as pkl
import shutil
import tarfile
from abc import ABCMeta, abstractmethod
from collections import OrderedDict, namedtuple
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import urlparse
from zipfile import ZipFile, is_zipfile

import numpy as np
import torch
from filelock import FileLock
from mmf.utils.configuration import get_mmf_cache_dir
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.batchnorm import BatchNorm2d
from torchvision.ops import RoIPool
from torchvision.ops.boxes import batched_nms


mmf_cache_home = get_mmf_cache_dir()

default_cache_path = os.path.join(mmf_cache_home, "transformers")
PYTORCH_PRETRAINED_BERT_CACHE = os.getenv(
    "PYTORCH_PRETRAINED_BERT_CACHE", default_cache_path
)
PYTORCH_TRANSFORMERS_CACHE = os.getenv(
    "PYTORCH_TRANSFORMERS_CACHE", PYTORCH_PRETRAINED_BERT_CACHE
)
TRANSFORMERS_CACHE = os.getenv("TRANSFORMERS_CACHE", PYTORCH_TRANSFORMERS_CACHE)
WEIGHTS_NAME = "pytorch_model.bin"


def load_checkpoint(ckp):
    r = OrderedDict()
    with open(ckp, "rb") as f:
        ckp = pkl.load(f)["model"]
    for k in copy.deepcopy(list(ckp.keys())):
        v = ckp.pop(k)
        if isinstance(v, np.ndarray):
            v = torch.tensor(v)
        else:
            assert isinstance(v, torch.tensor), type(v)
        r[k] = v
    return r


# TODO use mmf's download functionality
def cached_path(
    url_or_filename,
    cache_dir=None,
    force_download=False,
    proxies=None,
    resume_download=False,
    user_agent=None,
    extract_compressed_file=False,
    force_extract=False,
    local_files_only=False,
):
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    if isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    if os.path.exists(url_or_filename):
        # File, and it exists.
        output_path = url_or_filename
    elif urlparse(url_or_filename).scheme == "":
        # File, but it doesn't exist.
        raise OSError(f"file {url_or_filename} not found")
    else:
        # Something unknown
        raise ValueError(
            f"unable to parse {url_or_filename} as a URL or as a local path"
        )

    if extract_compressed_file:
        if not is_zipfile(output_path) and not tarfile.is_tarfile(output_path):
            return output_path

        # Path where we extract compressed archives
        # We avoid '.' in dir name and add "-extracted"
        # at the end: "./model.zip" => "./model-zip-extracted/"
        output_dir, output_file = os.path.split(output_path)
        output_extract_dir_name = output_file.replace(".", "-") + "-extracted"
        output_path_extracted = os.path.join(output_dir, output_extract_dir_name)

        if (
            os.path.isdir(output_path_extracted)
            and os.listdir(output_path_extracted)
            and not force_extract
        ):
            return output_path_extracted

        # Prevent parallel extractions
        lock_path = output_path + ".lock"
        with FileLock(lock_path):
            shutil.rmtree(output_path_extracted, ignore_errors=True)
            os.makedirs(output_path_extracted)
            if is_zipfile(output_path):
                with ZipFile(output_path, "r") as zip_file:
                    zip_file.extractall(output_path_extracted)
                    zip_file.close()
            elif tarfile.is_tarfile(output_path):
                tar_file = tarfile.open(output_path)
                tar_file.extractall(output_path_extracted)
                tar_file.close()
            else:
                raise OSError(
                    f"Archive format of {output_path} could not be identified"
                )

        return output_path_extracted

    return output_path


# Helper Functions
def _clip_box(tensor, box_size: Tuple[int, int]):
    assert torch.isfinite(tensor).all(), "Box tensor contains infinite or NaN!"
    h, w = box_size
    tensor[:, 0].clamp_(min=0, max=w)
    tensor[:, 1].clamp_(min=0, max=h)
    tensor[:, 2].clamp_(min=0, max=w)
    tensor[:, 3].clamp_(min=0, max=h)


def _nonempty_boxes(box, threshold: float = 0.0) -> torch.Tensor:
    widths = box[:, 2] - box[:, 0]
    heights = box[:, 3] - box[:, 1]
    keep = (widths > threshold) & (heights > threshold)
    return keep


def get_norm(norm, out_channels):
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": BatchNorm2d,
            "GN": lambda channels: nn.GroupNorm(32, channels),
            "nnSyncBN": nn.SyncBatchNorm,  # keep for debugging
            "": lambda x: x,
        }[norm]
    return norm(out_channels)


def _create_grid_offsets(size: List[int], stride: int, offset: float, device):

    grid_height, grid_width = size
    shifts_x = torch.arange(
        offset * stride,
        grid_width * stride,
        step=stride,
        dtype=torch.float32,
        device=device,
    )
    shifts_y = torch.arange(
        offset * stride,
        grid_height * stride,
        step=stride,
        dtype=torch.float32,
        device=device,
    )

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    return shift_x, shift_y


def build_backbone(cfg):
    input_shape = ShapeSpec(channels=len(cfg.model.pixel_mean))
    norm = cfg.resnets.norm
    stem = BasicStem(
        in_channels=input_shape.channels,
        out_channels=cfg.resnets.stem_out_channels,
        norm=norm,
        caffe_maxpool=cfg.model.max_pool,
    )
    freeze_at = cfg.backbone.freeze_at

    if freeze_at >= 1:
        for p in stem.parameters():
            p.requires_grad = False

    out_features = cfg.resnets.out_features
    depth = cfg.resnets.depth
    num_groups = cfg.resnets.num_groups
    width_per_group = cfg.resnets.width_per_group
    bottleneck_channels = num_groups * width_per_group
    in_channels = cfg.resnets.stem_out_channels
    out_channels = cfg.resnets.res2_out_channels
    stride_in_1x1 = cfg.resnets.stride_in_1x1
    res5_dilation = cfg.resnets.res5_dilation
    assert res5_dilation in {1, 2}, f"res5_dilation cannot be {res5_dilation}."

    num_blocks_per_stage = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}[
        depth
    ]

    stages = []
    out_stage_idx = [
        {"res2": 2, "res3": 3, "res4": 4, "res5": 5}[f] for f in out_features
    ]
    max_stage_idx = max(out_stage_idx)
    for idx, stage_idx in enumerate(range(2, max_stage_idx + 1)):
        dilation = res5_dilation if stage_idx == 5 else 1
        first_stride = 1 if idx == 0 or (stage_idx == 5 and dilation == 2) else 2
        stage_kargs = {
            "num_blocks": num_blocks_per_stage[idx],
            "first_stride": first_stride,
            "in_channels": in_channels,
            "bottleneck_channels": bottleneck_channels,
            "out_channels": out_channels,
            "num_groups": num_groups,
            "norm": norm,
            "stride_in_1x1": stride_in_1x1,
            "dilation": dilation,
        }

        stage_kargs["block_class"] = BottleneckBlock
        blocks = ResNet.make_stage(**stage_kargs)
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2

        if freeze_at >= stage_idx:
            for block in blocks:
                block.freeze()
        stages.append(blocks)

    return ResNet(stem, stages, out_features=out_features)


def find_top_rpn_proposals(
    proposals,
    pred_objectness_logits,
    images,
    image_sizes,
    nms_thresh,
    pre_nms_topk,
    post_nms_topk,
    min_box_side_len,
    training,
):
    """Args:
        proposals (list[Tensor]): (L, N, Hi*Wi*A, 4).
        pred_objectness_logits: tensors of length L.
        nms_thresh (float): IoU threshold to use for NMS
        pre_nms_topk (int): before nms
        post_nms_topk (int): after nms
        min_box_side_len (float): minimum proposal box side
        training (bool): True if proposals are to be used in training,
    Returns:
        results (List[Dict]): stores post_nms_topk object proposals for image i.
    """
    num_images = len(images)
    device = proposals[0].device

    # 1. Select top-k anchor for every level and every image
    topk_scores = []  # #lvl Tensor, each of shape N x topk
    topk_proposals = []
    level_ids = []  # #lvl Tensor, each of shape (topk,)
    batch_idx = torch.arange(num_images, device=device)
    for level_id, proposals_i, logits_i in zip(
        itertools.count(), proposals, pred_objectness_logits
    ):
        Hi_Wi_A = logits_i.shape[1]
        num_proposals_i = min(pre_nms_topk, Hi_Wi_A)

        # sort is faster than topk (https://github.com/pytorch/pytorch/issues/22812)
        # topk_scores_i, topk_idx = logits_i.topk(num_proposals_i, dim=1)
        logits_i, idx = logits_i.sort(descending=True, dim=1)
        topk_scores_i = logits_i[batch_idx, :num_proposals_i]
        topk_idx = idx[batch_idx, :num_proposals_i]

        # each is N x topk
        topk_proposals_i = proposals_i[batch_idx[:, None], topk_idx]  # N x topk x 4

        topk_proposals.append(topk_proposals_i)
        topk_scores.append(topk_scores_i)
        level_ids.append(
            torch.full((num_proposals_i,), level_id, dtype=torch.int64, device=device)
        )

    # 2. Concat all levels together
    topk_scores = torch.cat(topk_scores, dim=1)
    topk_proposals = torch.cat(topk_proposals, dim=1)
    level_ids = torch.cat(level_ids, dim=0)

    # if I change to batched_nms, I wonder if this will make a difference
    # 3. For each image, run a per-level NMS, and choose topk results.
    results = []
    for n, image_size in enumerate(image_sizes):
        boxes = topk_proposals[n]
        scores_per_img = topk_scores[n]
        # I will have to take a look at the boxes clip method
        _clip_box(boxes, image_size)
        # filter empty boxes
        keep = _nonempty_boxes(boxes, threshold=min_box_side_len)
        lvl = level_ids
        if keep.sum().item() != len(boxes):
            boxes, scores_per_img, lvl = (
                boxes[keep],
                scores_per_img[keep],
                level_ids[keep],
            )

        keep = batched_nms(boxes, scores_per_img, lvl, nms_thresh)
        keep = keep[:post_nms_topk]

        res = (boxes[keep], scores_per_img[keep])
        results.append(res)

    # I wonder if it would be possible for me to pad all these things.
    return results


def subsample_labels(labels, num_samples, positive_fraction, bg_label):
    """
    Returns:
        pos_idx, neg_idx (Tensor):
            1D vector of indices. The total length of both is `num_samples` or fewer.
    """
    positive = torch.nonzero((labels != -1) & (labels != bg_label)).squeeze(1)
    negative = torch.nonzero(labels == bg_label).squeeze(1)

    num_pos = int(num_samples * positive_fraction)
    # protect against not enough positive examples
    num_pos = min(positive.numel(), num_pos)
    num_neg = num_samples - num_pos
    # protect against not enough negative examples
    num_neg = min(negative.numel(), num_neg)

    # randomly select positive and negative examples
    perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
    perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

    pos_idx = positive[perm1]
    neg_idx = negative[perm2]
    return pos_idx, neg_idx


def add_ground_truth_to_proposals(gt_boxes, proposals):
    raise NotImplementedError()


def add_ground_truth_to_proposals_single_image(gt_boxes, proposals):
    raise NotImplementedError()


def _fmt_box_list(box_tensor, batch_index: int):
    repeated_index = torch.full(
        (len(box_tensor), 1),
        batch_index,
        dtype=box_tensor.dtype,
        device=box_tensor.device,
    )
    return torch.cat((repeated_index, box_tensor), dim=1)


def convert_boxes_to_pooler_format(box_lists: List[torch.Tensor]):
    pooler_fmt_boxes = torch.cat(
        [_fmt_box_list(box_list, i) for i, box_list in enumerate(box_lists)], dim=0
    )
    return pooler_fmt_boxes


def assign_boxes_to_levels(
    box_lists: List[torch.Tensor],
    min_level: int,
    max_level: int,
    canonical_box_size: int,
    canonical_level: int,
):

    box_sizes = torch.sqrt(torch.cat([boxes.area() for boxes in box_lists]))
    # Eqn.(1) in FPN paper
    level_assignments = torch.floor(
        canonical_level + torch.log2(box_sizes / canonical_box_size + 1e-8)
    )
    # clamp level to (min, max), in case the box size is too large or too small
    # for the available feature maps
    level_assignments = torch.clamp(level_assignments, min=min_level, max=max_level)
    return level_assignments.to(torch.int64) - min_level


# Helper Classes
class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None


class ShapeSpec(namedtuple("_ShapeSpec", ["channels", "height", "width", "stride"])):
    def __new__(cls, *, channels=None, height=None, width=None, stride=None):
        return super().__new__(cls, channels, height, width, stride)


class Box2BoxTransform:
    """
    This R-CNN transformation scales the box's width and height
    by exp(dw), exp(dh) and shifts a box's center by the offset
    (dx * width, dy * height).
    """

    def __init__(
        self, weights: Tuple[float, float, float, float], scale_clamp: float = None
    ):
        """
        Args:
            weights (4-element tuple): Scaling factors that are applied to the
                (dx, dy, dw, dh) deltas. In Fast R-CNN, these were originally set
                such that the deltas have unit variance; now they are treated as
                hyperparameters of the system.
            scale_clamp (float): When predicting deltas, the predicted box scaling
                factors (dw and dh) are clamped such that they are <= scale_clamp.
        """
        self.weights = weights
        if scale_clamp is not None:
            self.scale_clamp = scale_clamp
        else:
            """
            Value for clamping large dw and dh predictions.
            The heuristic is that we clamp such that dw and dh are no larger
            than what would transform a 16px box into a 1000px box
            (based on a small anchor, 16px, and a typical image size, 1000px).
            """
            self.scale_clamp = math.log(1000.0 / 16)

    def get_deltas(self, src_boxes, target_boxes):
        """
        Get box regression transformation deltas (dx, dy, dw, dh) that can be used
        to transform the `src_boxes` into the `target_boxes`. That is, the relation
        ``target_boxes == self.apply_deltas(deltas, src_boxes)`` is true (unless
        any delta is too large and is clamped).
        Args:
            src_boxes (Tensor): source boxes, e.g., object proposals
            target_boxes (Tensor): target of the transformation, e.g., ground-truth
                boxes.
        """
        assert isinstance(src_boxes, torch.Tensor), type(src_boxes)
        assert isinstance(target_boxes, torch.Tensor), type(target_boxes)

        src_widths = src_boxes[:, 2] - src_boxes[:, 0]
        src_heights = src_boxes[:, 3] - src_boxes[:, 1]
        src_ctr_x = src_boxes[:, 0] + 0.5 * src_widths
        src_ctr_y = src_boxes[:, 1] + 0.5 * src_heights

        target_widths = target_boxes[:, 2] - target_boxes[:, 0]
        target_heights = target_boxes[:, 3] - target_boxes[:, 1]
        target_ctr_x = target_boxes[:, 0] + 0.5 * target_widths
        target_ctr_y = target_boxes[:, 1] + 0.5 * target_heights

        wx, wy, ww, wh = self.weights
        dx = wx * (target_ctr_x - src_ctr_x) / src_widths
        dy = wy * (target_ctr_y - src_ctr_y) / src_heights
        dw = ww * torch.log(target_widths / src_widths)
        dh = wh * torch.log(target_heights / src_heights)

        deltas = torch.stack((dx, dy, dw, dh), dim=1)
        assert (
            (src_widths > 0).all().item()
        ), "Input boxes to Box2BoxTransform are not valid!"
        return deltas

    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.
        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h  # y2
        return pred_boxes


class Matcher:
    """
    This class assigns to each predicted "element" (e.g., a box) a ground-truth
    element. Each predicted element will have exactly zero or one matches; each
    ground-truth element may be matched to zero or more predicted elements.
    The matching is determined by the MxN match_quality_matrix, that characterizes
    how well each (ground-truth, prediction)-pair match each other. For example,
    if the elements are boxes, this matrix may contain box intersection-over-union
    overlap values.
    The matcher returns (a) a vector of length N containing the index of the
    ground-truth element m in [0, M) that matches to prediction n in [0, N).
    (b) a vector of length N containing the labels for each prediction.
    """

    def __init__(
        self,
        thresholds: List[float],
        labels: List[int],
        allow_low_quality_matches: bool = False,
    ):
        """
        Args:
            thresholds (list): a list of thresholds used to stratify predictions
                into levels.
            labels (list): a list of values to label predictions belonging at
                each level. A label can be one of {-1, 0, 1} signifying
                {ignore, negative class, positive class}, respectively.
            allow_low_quality_matches (bool): if True, produce additional matches or
                predictions with maximum match quality lower than high_threshold.
                For example, thresholds = [0.3, 0.5] labels = [0, -1, 1] All predictions
                with iou < 0.3 will be marked with 0 and
                thus will be considered as false positives while training. All
                predictions with 0.3 <= iou < 0.5 will be marked with -1 and
                thus will be ignored. All predictions with 0.5 <= iou will be marked
                with 1 and thus will be considered as true positives.
        """
        thresholds = thresholds[:]
        assert thresholds[0] > 0
        thresholds.insert(0, -float("inf"))
        thresholds.append(float("inf"))
        assert all(
            [low <= high for (low, high) in zip(thresholds[:-1], thresholds[1:])]
        )
        assert all([label_i in [-1, 0, 1] for label_i in labels])
        assert len(labels) == len(thresholds) - 1
        self.thresholds = thresholds
        self.labels = labels
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        """
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
            pairwise quality between M ground-truth elements and N predicted
                elements. All elements must be >= 0 (due to the us of `torch.nonzero`
                for selecting indices in :meth:`set_low_quality_matches_`).
        Returns:
            matches (Tensor[int64]): a vector of length N, where matches[i]
            is a matched ground-truth index in [0, M)
            match_labels (Tensor[int8]): a vector of length N, where pred_labels[i]
            indicates true or false positive or ignored
        """
        assert match_quality_matrix.dim() == 2
        if match_quality_matrix.numel() == 0:
            default_matches = match_quality_matrix.new_full(
                (match_quality_matrix.size(1),), 0, dtype=torch.int64
            )
            # When no gt boxes exist, we define IOU = 0 and therefore set labels
            # to `self.labels[0]`, which usually defaults to background class 0
            # To choose to ignore instead,
            # can make labels=[-1,0,-1,1] + set appropriate thresholds
            default_match_labels = match_quality_matrix.new_full(
                (match_quality_matrix.size(1),), self.labels[0], dtype=torch.int8
            )
            return default_matches, default_match_labels

        assert torch.all(match_quality_matrix >= 0)

        # match_quality_matrix is M (gt) x N (predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction
        matched_vals, matches = match_quality_matrix.max(dim=0)

        match_labels = matches.new_full(matches.size(), 1, dtype=torch.int8)

        for (l, low, high) in zip(
            self.labels, self.thresholds[:-1], self.thresholds[1:]
        ):
            low_high = (matched_vals >= low) & (matched_vals < high)
            match_labels[low_high] = l

        if self.allow_low_quality_matches:
            self.set_low_quality_matches_(match_labels, match_quality_matrix)

        return matches, match_labels

    def set_low_quality_matches_(self, match_labels, match_quality_matrix):
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth G find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth G.
        This function implements the RPN assignment case (i)
        in Sec. 3.1.2 of Faster R-CNN.
        """
        # For each gt, find the prediction with which it has highest quality
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        # Find the highest quality match available, even if it is low, including ties.
        # Note that the matches qualities must be positive due to the use of
        # `torch.nonzero`.
        of_quality_inds = match_quality_matrix == highest_quality_foreach_gt[:, None]
        if of_quality_inds.dim() == 0:
            (_, pred_inds_with_highest_quality) = (
                of_quality_inds.unsqueeze(0).nonzero().unbind(1)
            )
        else:
            (_, pred_inds_with_highest_quality) = of_quality_inds.nonzero().unbind(1)
        match_labels[pred_inds_with_highest_quality] = 1


class RPNOutputs:
    def __init__(
        self,
        box2box_transform,
        anchor_matcher,
        batch_size_per_image,
        positive_fraction,
        images,
        pred_objectness_logits,
        pred_anchor_deltas,
        anchors,
        boundary_threshold=0,
        gt_boxes=None,
        smooth_l1_beta=0.0,
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform): :class:`Box2BoxTransform`
            instance for anchor-proposal transformations.
            anchor_matcher (Matcher): :class:`Matcher` instance for matching
            anchors to ground-truth boxes; used to determine training labels.
            batch_size_per_image (int): number of proposals to sample when training
            positive_fraction (float): target fraction of sampled proposals that
            should be positive
            images (ImageList): :class:`ImageList` instance representing N input images
            pred_objectness_logits (list[Tensor]): A list of L elements. Element i
            is a tensor of shape (N, A, Hi, W)
            pred_anchor_deltas (list[Tensor]): A list of L elements. Element i is a
            tensor of shape (N, A*4, Hi, Wi)
            anchors (list[torch.Tensor]): nested list of boxes. anchors[i][j] at (n, l)
            stores anchor array for feature map l
            boundary_threshold (int): if >= 0, then anchors that extend beyond the image
            boundary by more than boundary_thresh are not used in training.
            gt_boxes (list[Boxes], optional): A list of N elements.
            smooth_l1_beta (float): The transition point between L1 and L2 lossn. When
            set to 0, the loss becomes L1. When +inf, it is ignored
        """
        self.box2box_transform = box2box_transform
        self.anchor_matcher = anchor_matcher
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.pred_objectness_logits = pred_objectness_logits
        self.pred_anchor_deltas = pred_anchor_deltas

        self.anchors = anchors
        self.gt_boxes = gt_boxes
        self.num_feature_maps = len(pred_objectness_logits)
        self.num_images = len(images)
        self.boundary_threshold = boundary_threshold
        self.smooth_l1_beta = smooth_l1_beta

    def _get_ground_truth(self):
        raise NotImplementedError()

    def predict_proposals(self):
        # pred_anchor_deltas: (L, N, ? Hi, Wi)
        # anchors:(N, L, -1, B)
        # here we loop over specific feature map, NOT images
        proposals = []
        anchors = self.anchors.transpose(0, 1)
        for anchors_i, pred_anchor_deltas_i in zip(anchors, self.pred_anchor_deltas):
            B = anchors_i.size(-1)
            N, _, Hi, Wi = pred_anchor_deltas_i.shape
            anchors_i = anchors_i.flatten(start_dim=0, end_dim=1)
            pred_anchor_deltas_i = (
                pred_anchor_deltas_i.view(N, -1, B, Hi, Wi)
                .permute(0, 3, 4, 1, 2)
                .reshape(-1, B)
            )
            proposals_i = self.box2box_transform.apply_deltas(
                pred_anchor_deltas_i, anchors_i
            )
            # Append feature map proposals with shape (N, Hi*Wi*A, B)
            proposals.append(proposals_i.view(N, -1, B))
        proposals = torch.stack(proposals)
        return proposals

    def predict_objectness_logits(self):
        """
        Returns:
            pred_objectness_logits (list[Tensor]) -> (N, Hi*Wi*A).
        """
        pred_objectness_logits = [
            # Reshape: (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).reshape(self.num_images, -1)
            for score in self.pred_objectness_logits
        ]
        return pred_objectness_logits


# Main Classes
class Conv2d(torch.nn.Conv2d):
    def __init__(self, *args, **kwargs):
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        if x.numel() == 0 and self.training:
            assert not isinstance(self.norm, torch.nn.SyncBatchNorm)
        if x.numel() == 0:
            assert not isinstance(self.norm, torch.nn.GroupNorm)
            output_shape = [
                (i + 2 * p - (di * (k - 1) + 1)) // s + 1
                for i, p, di, k, s in zip(
                    x.shape[-2:],
                    self.padding,
                    self.dilation,
                    self.kernel_size,
                    self.stride,
                )
            ]
            output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
            empty = _NewEmptyTensorOp.apply(x, output_shape)
            if self.training:
                _dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                return empty + _dummy
            else:
                return empty

        x = super().forward(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class LastLevelMaxPool(nn.Module):
    """
    This module is used in the original FPN to generate a downsampled P6
    feature from P5.
    """

    def __init__(self):
        super().__init__()
        self.num_levels = 1
        self.in_feature = "p5"

    def forward(self, x):
        return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]


class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7
    from C5 feature.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.num_levels = 2
        self.in_feature = "res5"
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)

    def forward(self, c5):
        p6 = self.p6(c5)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]


class BasicStem(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, norm="BN", caffe_maxpool=False):
        super().__init__()
        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            norm=get_norm(norm, out_channels),
        )
        self.caffe_maxpool = caffe_maxpool
        # use pad 1 instead of pad zero

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu_(x)
        if self.caffe_maxpool:
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=0, ceil_mode=True)
        else:
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x

    @property
    def out_channels(self):
        return self.conv1.out_channels

    @property
    def stride(self):
        return 4  # = stride 2 conv -> stride 2 max pool


class ResNetBlockBase(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        return self


class BottleneckBlock(ResNetBlockBase):
    def __init__(
        self,
        in_channels,
        out_channels,
        bottleneck_channels,
        stride=1,
        num_groups=1,
        norm="BN",
        stride_in_1x1=False,
        dilation=1,
    ):
        super().__init__(in_channels, out_channels, stride)

        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
        else:
            self.shortcut = None

        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
        # stride in the 3x3 conv
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
            norm=get_norm(norm, bottleneck_channels),
        )

        self.conv2 = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=1 * dilation,
            bias=False,
            groups=num_groups,
            dilation=dilation,
            norm=get_norm(norm, bottleneck_channels),
        )

        self.conv3 = Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)

        out = self.conv2(out)
        out = F.relu_(out)

        out = self.conv3(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu_(out)
        return out


class Backbone(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self):
        pass

    @property
    def size_divisibility(self):
        """
        Some backbones require the input height and width to be divisible
        by a specific integer. This is
        typically true for encoder / decoder type networks with lateral
        connection (e.g., FPN) for which feature maps need to match
        dimension in the "bottom up" and "top down" paths. Set to 0 if
        no specific input size divisibility is required.
        """
        return 0

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }

    @property
    def out_features(self):
        """deprecated"""
        return self._out_features

    @property
    def out_feature_strides(self):
        """deprecated"""
        return {f: self._out_feature_strides[f] for f in self._out_features}

    @property
    def out_feature_channels(self):
        """deprecated"""
        return {f: self._out_feature_channels[f] for f in self._out_features}


class ResNet(Backbone):
    def __init__(self, stem, stages, num_classes=None, out_features=None):
        """
        Args:
            stem (nn.Module): a stem module
            stages (list[list[ResNetBlock]]): several (typically 4) stages,
            each contains multiple :class:`ResNetBlockBase`.
            num_classes (None or int): if None, will not perform classification.
            out_features (list[str]): name of the layers whose outputs should
            be returned in forward. Can be anything in:
            "stem", "linear", or "res2" ... If None, will return the output
            of the last layer.
        """
        super().__init__()
        self.stem = stem
        self.num_classes = num_classes

        current_stride = self.stem.stride
        self._out_feature_strides = {"stem": current_stride}
        self._out_feature_channels = {"stem": self.stem.out_channels}

        self.stages_and_names = []
        for i, blocks in enumerate(stages):
            for block in blocks:
                assert isinstance(block, ResNetBlockBase), block
                curr_channels = block.out_channels
            stage = nn.Sequential(*blocks)
            name = "res" + str(i + 2)
            self.add_module(name, stage)
            self.stages_and_names.append((stage, name))
            self._out_feature_strides[name] = current_stride = int(
                current_stride * np.prod([k.stride for k in blocks])
            )
            self._out_feature_channels[name] = blocks[-1].out_channels

        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(curr_channels, num_classes)

            # Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
            # "The 1000-way fully-connected layer is initialized by
            # drawing weights from a zero-mean Gaussian with std of 0.01."
            nn.init.normal_(self.linear.weight, stddev=0.01)
            name = "linear"

        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {}".format(
                ", ".join(children)
            )

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for stage, name in self.stages_and_names:
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        if self.num_classes is not None:
            x = self.avgpool(x)
            x = self.linear(x)
            if "linear" in self._out_features:
                outputs["linear"] = x
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }

    @staticmethod
    def make_stage(
        block_class,
        num_blocks,
        first_stride=None,
        *,
        in_channels,
        out_channels,
        **kwargs,
    ):
        """
        Usually, layers that produce the same feature map spatial size
        are defined as one "stage".
        Under such definition, stride_per_block[1:] should all be 1.
        """
        if first_stride is not None:
            assert "stride" not in kwargs and "stride_per_block" not in kwargs
            kwargs["stride_per_block"] = [first_stride] + [1] * (num_blocks - 1)
        blocks = []
        for i in range(num_blocks):
            curr_kwargs = {}
            for k, v in kwargs.items():
                if k.endswith("_per_block"):
                    assert len(v) == num_blocks, (
                        f"Argument '{k}' of make_stage should have the "
                        f"same length as num_blocks={num_blocks}."
                    )
                    newk = k[: -len("_per_block")]
                    assert (
                        newk not in kwargs
                    ), f"Cannot call make_stage with both {k} and {newk}!"
                    curr_kwargs[newk] = v[i]
                else:
                    curr_kwargs[k] = v

            blocks.append(
                block_class(
                    in_channels=in_channels, out_channels=out_channels, **curr_kwargs
                )
            )
            in_channels = out_channels

        return blocks


class ROIPooler(nn.Module):
    """
    Region of interest feature map pooler that supports pooling from one or more
    feature maps.
    """

    def __init__(
        self,
        output_size,
        scales,
        sampling_ratio,
        canonical_box_size=224,
        canonical_level=4,
    ):
        super().__init__()
        # assumption that stride is a power of 2.
        min_level = -math.log2(scales[0])
        max_level = -math.log2(scales[-1])

        # a bunch of testing
        assert math.isclose(min_level, int(min_level)) and math.isclose(
            max_level, int(max_level)
        )
        assert len(scales) == max_level - min_level + 1, "not pyramid"
        assert 0 < min_level and min_level <= max_level
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        assert (
            len(output_size) == 2
            and isinstance(output_size[0], int)
            and isinstance(output_size[1], int)
        )
        if len(scales) > 1:
            assert min_level <= canonical_level and canonical_level <= max_level
        assert canonical_box_size > 0

        self.output_size = output_size
        self.min_level = int(min_level)
        self.max_level = int(max_level)
        self.level_poolers = nn.ModuleList(
            RoIPool(output_size, spatial_scale=scale) for scale in scales
        )
        self.canonical_level = canonical_level
        self.canonical_box_size = canonical_box_size

    def forward(self, feature_maps, boxes):
        """
        Args:
            feature_maps: List[torch.Tensor(N,C,W,H)]
            box_lists: list[torch.Tensor])
        Returns:
            A tensor of shape(N*B, Channels, output_size, output_size)
        """
        x = [v for v in feature_maps.values()]
        num_level_assignments = len(self.level_poolers)
        assert len(x) == num_level_assignments and len(boxes) == x[0].size(0)

        pooler_fmt_boxes = convert_boxes_to_pooler_format(boxes)

        if num_level_assignments == 1:
            return self.level_poolers[0](x[0], pooler_fmt_boxes)

        level_assignments = assign_boxes_to_levels(
            boxes,
            self.min_level,
            self.max_level,
            self.canonical_box_size,
            self.canonical_level,
        )

        num_boxes = len(pooler_fmt_boxes)
        num_channels = x[0].shape[1]
        output_size = self.output_size[0]

        dtype, device = x[0].dtype, x[0].device
        output = torch.zeros(
            (num_boxes, num_channels, output_size, output_size),
            dtype=dtype,
            device=device,
        )

        for level, (x_level, pooler) in enumerate(zip(x, self.level_poolers)):
            inds = torch.nonzero(level_assignments == level).squeeze(1)
            pooler_fmt_boxes_level = pooler_fmt_boxes[inds]
            output[inds] = pooler(x_level, pooler_fmt_boxes_level)

        return output


class Res5ROIHeads(nn.Module):
    """
    ROIHeads perform all per-region computation in an R-CNN.
    It contains logic of cropping the regions, extract per-region features
    (by the res-5 block in this case), and make per-region predictions.
    """

    def __init__(self, cfg, input_shape):
        super().__init__()
        self.batch_size_per_image = cfg.rpn.batch_size_per_image
        self.positive_sample_fraction = cfg.roi_heads.positive_fraction
        self.in_features = cfg.roi_heads.in_features
        self.num_classes = cfg.roi_heads.num_classes
        self.proposal_append_gt = cfg.roi_heads.proposal_append_gt
        self.feature_strides = {k: v.stride for k, v in input_shape.items()}
        self.feature_channels = {k: v.channels for k, v in input_shape.items()}
        self.cls_agnostic_bbox_reg = cfg.roi_box_head.cls_agnostic_bbox_reg
        self.stage_channel_factor = 2 ** 3  # res5 is 8x res2
        self.out_channels = cfg.resnets.res2_out_channels * self.stage_channel_factor

        # self.proposal_matcher = Matcher(
        #     cfg.ROI_HEADS.IOU_THRESHOLDS,
        #     cfg.ROI_HEADS.IOU_LABELS,
        #     allow_low_quality_matches=False,
        # )

        pooler_resolution = cfg.roi_box_head.pooler_resolution
        pooler_scales = (1.0 / self.feature_strides[self.in_features[0]],)
        sampling_ratio = cfg.roi_box_head.pooler_sampling_ratio
        res5_halve = cfg.roi_box_head.res5halve

        self.pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
        )

        self.res5 = self._build_res5_block(cfg)
        if not res5_halve:
            """
            Modifications for VG in RoI heads:
            1. Change the stride of conv1 and shortcut in Res5.Block1 from 2 to 1
            2. Modifying all conv2 with (padding: 1 --> 2) and (dilation: 1 --> 2)
            """
            self.res5[0].conv1.stride = (1, 1)
            self.res5[0].shortcut.stride = (1, 1)
            for i in range(3):
                self.res5[i].conv2.padding = (2, 2)
                self.res5[i].conv2.dilation = (2, 2)

    def _build_res5_block(self, cfg):
        stage_channel_factor = self.stage_channel_factor  # res5 is 8x res2
        num_groups = cfg.resnets.num_groups
        width_per_group = cfg.resnets.width_per_group
        bottleneck_channels = num_groups * width_per_group * stage_channel_factor
        out_channels = self.out_channels
        stride_in_1x1 = cfg.resnets.stride_in_1x1
        norm = cfg.resnets.norm

        blocks = ResNet.make_stage(
            BottleneckBlock,
            3,
            first_stride=2,
            in_channels=out_channels // 2,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            norm=norm,
            stride_in_1x1=stride_in_1x1,
        )
        return nn.Sequential(*blocks)

    def _shared_roi_transform(self, features, boxes):
        x = self.pooler(features, boxes)
        return self.res5(x)

    def forward(self, features, proposal_boxes, gt_boxes=None):
        if self.training:
            """
            see https://github.com/airsplay/py-bottom-up-attention/\
                    blob/master/detectron2/modeling/roi_heads/roi_heads.py
            """
            raise NotImplementedError()

        assert not proposal_boxes[0].requires_grad
        box_features = self._shared_roi_transform(features, proposal_boxes)
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        return feature_pooled


class AnchorGenerator(nn.Module):
    """
    For a set of image sizes and feature maps, computes a set of anchors.
    """

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        sizes = list(cfg.anchor_generator.sizes)
        aspect_ratios = list(cfg.anchor_generator.aspect_ratios)
        self.strides = [x.stride for x in input_shape]
        self.offset = cfg.anchor_generator.offset
        assert 0.0 <= self.offset < 1.0, self.offset

        """
        sizes (list[list[int]]): sizes[i] is the list of anchor sizes for feat map i
            1. given in absolute lengths in units of the input image;
            2. they do not dynamically scale if the input image size changes.
        aspect_ratios (list[list[float]])
        strides (list[int]): stride of each input feature.
        """

        self.num_features = len(self.strides)
        self.cell_anchors = nn.ParameterList(
            self._calculate_anchors(sizes, aspect_ratios)
        )
        self._spacial_feat_dim = 4

    def _calculate_anchors(self, sizes, aspect_ratios):
        # If one size (or aspect ratio) is specified and there are multiple feature
        # maps, then we "broadcast" anchors of that single size (or aspect ratio)
        if len(sizes) == 1:
            sizes *= self.num_features
        if len(aspect_ratios) == 1:
            aspect_ratios *= self.num_features
        assert self.num_features == len(sizes)
        assert self.num_features == len(aspect_ratios)

        cell_anchors = [
            self.generate_cell_anchors(s, a).float()
            for s, a in zip(sizes, aspect_ratios)
        ]

        return cell_anchors

    @property
    def box_dim(self):
        return self._spacial_feat_dim

    @property
    def num_cell_anchors(self):
        """
        Returns:
            list[int]: Each int is the number of anchors at every pixel
            location, on that feature map.
        """
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    def grid_anchors(self, grid_sizes):
        anchors = []
        for (size, stride, base_anchors) in zip(
            grid_sizes, self.strides, self.cell_anchors
        ):
            shift_x, shift_y = _create_grid_offsets(
                size, stride, self.offset, base_anchors.device
            )
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            anchors.append(
                (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)
            )

        return anchors

    def generate_cell_anchors(
        self, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)
    ):
        """
        anchors are continuous geometric rectangles
        centered on one feature map point sample.
        We can later build the set of anchors
        for the entire feature map by tiling these tensors
        """

        anchors = []
        for size in sizes:
            area = size ** 2.0
            for aspect_ratio in aspect_ratios:
                w = math.sqrt(area / aspect_ratio)
                h = aspect_ratio * w
                x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
                anchors.append([x0, y0, x1, y1])
        return nn.Parameter(torch.Tensor(anchors))

    def forward(self, features):
        """
        Args:
            features List[torch.Tensor]: list of feature maps on which to
            generate anchors.
        Returns:
            torch.Tensor: a list of #image elements.
        """
        num_images = features[0].size(0)
        grid_sizes = [feature_map.shape[-2:] for feature_map in features]
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes)
        anchors_over_all_feature_maps = torch.stack(anchors_over_all_feature_maps)
        return anchors_over_all_feature_maps.unsqueeze(0).repeat_interleave(
            num_images, dim=0
        )


class RPNHead(nn.Module):
    """
    RPN classification and regression heads. Uses a 3x3 conv to produce a shared
    hidden state from which one 1x1 conv predicts objectness logits for each anchor
    and a second 1x1 conv predicts bounding-box deltas specifying how to deform
    each anchor into an object proposal.
    """

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()

        # Standard RPN is shared across levels:
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        anchor_generator = AnchorGenerator(cfg, input_shape)
        num_cell_anchors = anchor_generator.num_cell_anchors
        box_dim = anchor_generator.box_dim
        assert (
            len(set(num_cell_anchors)) == 1
        ), "Each level must have the same number of cell anchors"
        num_cell_anchors = num_cell_anchors[0]

        if cfg.proposal_generator.hidden_channels == -1:
            hid_channels = in_channels
        else:
            hid_channels = cfg.proposal_generator.hidden_channels
            # Modifications for VG in RPN (modeling/proposal_generator/rpn.py)
            # Use hidden dim  instead fo the same dim as Res4 (in_channels)

        # 3x3 conv for the hidden representation
        self.conv = nn.Conv2d(
            in_channels, hid_channels, kernel_size=3, stride=1, padding=1
        )
        # 1x1 conv for predicting objectness logits
        self.objectness_logits = nn.Conv2d(
            hid_channels, num_cell_anchors, kernel_size=1, stride=1
        )
        # 1x1 conv for predicting box2box transform deltas
        self.anchor_deltas = nn.Conv2d(
            hid_channels, num_cell_anchors * box_dim, kernel_size=1, stride=1
        )

        for layer in [self.conv, self.objectness_logits, self.anchor_deltas]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def forward(self, features):
        """
        Args:
            features (list[Tensor]): list of feature maps
        """
        pred_objectness_logits = []
        pred_anchor_deltas = []
        for x in features:
            t = F.relu(self.conv(x))
            pred_objectness_logits.append(self.objectness_logits(t))
            pred_anchor_deltas.append(self.anchor_deltas(t))
        return pred_objectness_logits, pred_anchor_deltas


class RPN(nn.Module):
    """
    Region Proposal Network, introduced by the Faster R-CNN paper.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        self.min_box_side_len = cfg.proposal_generator.min_size
        self.in_features = cfg.rpn.in_features
        self.nms_thresh = cfg.rpn.nms_thresh
        self.batch_size_per_image = cfg.rpn.batch_size_per_image
        self.positive_fraction = cfg.rpn.positive_fraction
        self.smooth_l1_beta = cfg.rpn.smooth_l1_beta
        self.loss_weight = cfg.rpn.loss_weight

        self.pre_nms_topk = {
            True: cfg.rpn.pre_nms_topk_train,
            False: cfg.rpn.pre_nms_topk_test,
        }
        self.post_nms_topk = {
            True: cfg.rpn.post_nms_topk_train,
            False: cfg.rpn.post_nms_topk_test,
        }
        self.boundary_threshold = cfg.rpn.boundary_thresh

        self.anchor_generator = AnchorGenerator(
            cfg, [input_shape[f] for f in self.in_features]
        )
        self.box2box_transform = Box2BoxTransform(weights=cfg.rpn.bbox_reg_weights)
        self.anchor_matcher = Matcher(
            cfg.rpn.iou_thresholds, cfg.rpn.iou_labels, allow_low_quality_matches=True
        )
        self.rpn_head = RPNHead(cfg, [input_shape[f] for f in self.in_features])

    def training(self, images, image_shapes, features, gt_boxes):
        pass

    def inference(self, outputs, images, image_shapes, features, gt_boxes=None):
        outputs = find_top_rpn_proposals(
            outputs.predict_proposals(),
            outputs.predict_objectness_logits(),
            images,
            image_shapes,
            self.nms_thresh,
            self.pre_nms_topk[self.training],
            self.post_nms_topk[self.training],
            self.min_box_side_len,
            self.training,
        )

        results = []
        for img in outputs:
            im_boxes, img_box_logits = img
            img_box_logits, inds = img_box_logits.sort(descending=True)
            im_boxes = im_boxes[inds]
            results.append((im_boxes, img_box_logits))

        (proposal_boxes, logits) = tuple(map(list, zip(*results)))
        return proposal_boxes, logits

    def forward(self, images, image_shapes, features, gt_boxes=None):
        """
        Args:
            images (torch.Tensor): input images of length `N`
            features (dict[str: Tensor])
            gt_instances
        """
        # features is dict, key = block level, v = feature_map
        features = [features[f] for f in self.in_features]
        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
        anchors = self.anchor_generator(features)
        outputs = RPNOutputs(
            self.box2box_transform,
            self.anchor_matcher,
            self.batch_size_per_image,
            self.positive_fraction,
            images,
            pred_objectness_logits,
            pred_anchor_deltas,
            anchors,
            self.boundary_threshold,
            gt_boxes,
            self.smooth_l1_beta,
        )
        # For RPN-only models, the proposals are the final output

        if self.training:
            raise NotImplementedError()
            return self.training(outputs, images, image_shapes, features, gt_boxes)
        else:
            return self.inference(outputs, images, image_shapes, features, gt_boxes)


class GeneralizedRCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.model.device)
        self.backbone = build_backbone(cfg)
        self.proposal_generator = RPN(cfg, self.backbone.output_shape())
        self.roi_heads = Res5ROIHeads(cfg, self.backbone.output_shape())
        self.to(self.device)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        state_dict = kwargs.pop("state_dict", None)
        cache_dir = kwargs.pop("cache_dir", None)
        from_tf = kwargs.pop("from_tf", False)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)

        # Load model
        if pretrained_model_name_or_path is not None:
            if os.path.isdir(pretrained_model_name_or_path):
                if os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
                ):
                    # Load from a PyTorch checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, WEIGHTS_NAME
                    )
                else:
                    raise OSError(
                        "Error no file named {} found in directory {} ".format(
                            WEIGHTS_NAME, pretrained_model_name_or_path
                        )
                    )
            elif os.path.isfile(pretrained_model_name_or_path):
                archive_file = pretrained_model_name_or_path
            elif os.path.isfile(pretrained_model_name_or_path + ".index"):
                assert (
                    from_tf
                ), "We found a TensorFlow checkpoint at {}, please set from_tf \
                to True to load from this checkpoint".format(
                    pretrained_model_name_or_path + ".index"
                )
                archive_file = pretrained_model_name_or_path + ".index"

            try:
                # Load from URL or cache if already cached
                resolved_archive_file = cached_path(
                    archive_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                )
                if resolved_archive_file is None:
                    raise OSError
            except OSError:
                msg = f"Can't load weights for '{pretrained_model_name_or_path}'."
                raise OSError(msg)

            if resolved_archive_file == archive_file:
                print(f"loading weights file {archive_file}")
            else:
                print(
                    f"loading weights file {archive_file} from cache at "
                    + f"{resolved_archive_file}"
                )
        else:
            resolved_archive_file = None

        # Instantiate model.
        model = cls(config)

        if state_dict is None:
            try:
                try:
                    state_dict = torch.load(resolved_archive_file, map_location="cpu")
                except Exception:
                    state_dict = load_checkpoint(resolved_archive_file)

            except Exception:
                raise OSError(
                    "Unable to load weights from pytorch checkpoint file. "
                    "If you tried to load a PyTorch model from a TF 2.0 "
                    + "checkpoint, please set from_tf=True. "
                )

        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        # Convert old format to new format if needed from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if "gamma" in key:
                new_key = key.replace("gamma", "weight")
            if "beta" in key:
                new_key = key.replace("beta", "bias")
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        statedict = model.state_dict().keys()
        keys = list(state_dict.keys())
        for key in keys:
            if key not in statedict:
                del state_dict[key]

        model_to_load = model
        model_to_load.load_state_dict(state_dict)

        if model.__class__.__name__ != model_to_load.__class__.__name__:
            base_model_state_dict = model_to_load.state_dict().keys()
            head_model_state_dict_without_base_prefix = [
                key.split(cls.base_model_prefix + ".")[-1]
                for key in model.state_dict().keys()
            ]
            missing_keys.extend(
                head_model_state_dict_without_base_prefix - base_model_state_dict
            )

        if len(unexpected_keys) > 0:
            print(
                "Some weights of the model checkpoint at "
                f"{pretrained_model_name_or_path} were not used when "
                f"initializing {model.__class__.__name__}: {unexpected_keys}\n"
                f"- This IS expected if you are initializing "
                f"{model.__class__.__name__} from the checkpoint of a model trained "
                "on another task or with another architecture (e.g. initializing a "
                "BertForSequenceClassification model from a BertForPreTraining model)."
                "\n- This IS NOT expected if you are initializing "
                f"{model.__class__.__name__} from the checkpoint of a model "
                "that you expect to be exactly identical (initializing a "
                "BertForSequenceClassification model from a "
                "BertForSequenceClassification model)."
            )
        else:
            print(
                f"All model checkpoint weights were used when initializing "
                + f"{model.__class__.__name__}.\n"
            )
        if len(missing_keys) > 0:
            print(
                f"Some weights of {model.__class__.__name__} were not initialized "
                + f"from the model checkpoint at {pretrained_model_name_or_path} "
                f"and are newly initialized: {missing_keys}\n"
                f"You should probably TRAIN this model on a down-stream task to be "
                + "able to use it for predictions and inference."
            )
        else:
            print(
                f"All the weights of {model.__class__.__name__} were initialized "
                + f"from the model checkpoint at {pretrained_model_name_or_path}.\n"
                f"If your task is similar to the task the model of the "
                + "checkpoint was trained on, "
                f"you can already use {model.__class__.__name__} for "
                + "predictions without further training."
            )
        if len(error_msgs) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    model.__class__.__name__, "\n\t".join(error_msgs)
                )
            )
        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()

        return model

    def forward(
        self,
        images,
        image_shapes,
        gt_boxes=None,
        proposals=None,
        scales_yx=None,
        **kwargs,
    ):
        """
        kwargs:
            max_detections (int), return_tensors {"np", "pt", None}, padding {None,
            "max_detections"}, pad_value (int), location = {"cuda", "cpu"}
        """
        if self.training:
            raise NotImplementedError()
        return self.inference(
            images=images,
            image_shapes=image_shapes,
            gt_boxes=gt_boxes,
            proposals=proposals,
            scales_yx=scales_yx,
            **kwargs,
        )

    @torch.no_grad()
    def inference(
        self,
        images,
        image_shapes,
        gt_boxes=None,
        proposals=None,
        scales_yx=None,
        **kwargs,
    ):
        # run images through backbone
        features = self.backbone(images)

        # generate proposals if none are available
        if proposals is None:
            proposal_boxes, _ = self.proposal_generator(
                images, image_shapes, features, gt_boxes
            )
        else:
            assert proposals is not None

        # pool object features from either gt_boxes, or from proposals
        feature_pooled = self.roi_heads(features, proposal_boxes, gt_boxes)

        preds_per_image = [p.size(0) for p in proposal_boxes]

        roi_features = feature_pooled.split(preds_per_image, dim=0)

        return roi_features
