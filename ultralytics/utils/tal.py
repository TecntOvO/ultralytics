# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn

from . import LOGGER
from .checks import check_version
from .metrics import bbox_iou, probiou
from .ops import xywhr2xyxyxyxy

TORCH_1_10 = check_version(torch.__version__, "1.10.0")


class TaskAlignedAssigner(nn.Module):
    """
    A task-aligned assigner for object detection.

    This class assigns ground-truth (gt) objects to anchors based on the task-aligned metric, which combines both
    classification and localization information.

    Attributes:
        topk (int): The number of top candidates to consider.
        num_classes (int): The number of object classes.
        alpha (float): The alpha parameter for the classification component of the task-aligned metric.
        beta (float): The beta parameter for the localization component of the task-aligned metric.
        eps (float): A small value to prevent division by zero.
    """

    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9):
        """Initialize a TaskAlignedAssigner object with customizable hyperparameters."""
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        Compute the task-aligned assignment. Reference code is available at
        https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py.

        Args:
            pd_scores (Tensor): shape(bs, num_total_anchors, num_classes) 预测的类别分数
            pd_bboxes (Tensor): shape(bs, num_total_anchors, 4) 预测的边界框坐标
            anc_points (Tensor): shape(num_total_anchors, 2) 锚点中心坐标
            gt_labels (Tensor): shape(bs, n_max_boxes, 1) 真实类别标签
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4) 真实边界框坐标
            mask_gt (Tensor): shape(bs, n_max_boxes, 1) 真实框存在的掩码

        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors) 锚点分配的类别标签
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4) 锚点分配的真实框坐标
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes) 锚点分配的类别分数
            fg_mask (Tensor): shape(bs, num_total_anchors) 前景（正样本）掩码
            target_gt_idx (Tensor): shape(bs, num_total_anchors) 锚点对应的真实框索引
        """
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]
        device = gt_bboxes.device

        if self.n_max_boxes == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.bg_idx),
               torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0]),
                torch.zeros_like(pd_scores[..., 0]),
            )

        try:
            return self._forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
        except torch.OutOfMemoryError:
            # Move tensors to CPU, compute, then move back to original device
            LOGGER.warning("WARNING: CUDA OutOfMemoryError in TaskAlignedAssigner, using CPU")
            cpu_tensors = [t.cpu() for t in (pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)]
            result = self._forward(*cpu_tensors)
            return tuple(t.to(device) for t in result)

    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        Compute the task-aligned assignment. Reference code is available at
        https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py.

        Args:
            pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            anc_points (Tensor): shape(num_total_anchors, 2)
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)

        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
            target_gt_idx (Tensor): shape(bs, num_total_anchors)
        """
        # 候选锚点筛选
        # mask_pos -> [batchsize, n_max_boxes, num_total_anchors] 表示每个正样本的候选锚点
        # align_metric -> [batchsize, n_max_boxes, num_total_anchors] 每个元素值为这个正样本下这个锚点的对齐指标
        # overlaps -> [batchsize, n_max_boxes, num_total_anchors] 每个元素值为这个正样本下这个锚框和对应正样本的框的iou
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )
        # 当一个锚点被分配给多个正样本时，选择与其具有最高IoU的那个真实框进行匹配
        # target_gt_idx -> [batchsize, num_total_anchors] 表示每个锚点最终分配的正样本的索引
        # fg_mask -> [batchsize, num_total_anchors] 1表示有对应正样本的锚框
        # mask_pos -> [batchsize, n_max_boxes, num_total_anchors] 表示每个图片每个正样本分配的锚点（不会有锚点同时分配到多个真实目标上）
        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)
        small_fg_mask = self.get_selected_box_mask(gt_bboxes, fg_mask, mask_gt, target_gt_idx, 32)

        # Assigned target
        # 为分配了真实目标的锚点，根据对应真实目标的数据为其分配对应的目标标签、目标边界框和目标得分
        # 根据 target_gt_idx 将真实标签和框映射到锚点
        # target_labels -> [bs, num_total_anchors] 分配的类别标签
        # target_bboxes -> [bs, num_anchors, 4] 分配的真实框坐标
        # target_scores -> [bs, num_anchors, num_classes] 分配的类别分数（one-hot 编码）
        # target_scores表示每个锚点对应的目标分数。锚点对应正样本的分类的分数为 1，其他为 0
        target_labels, target_bboxes, target_scores= self.get_targets(gt_labels, gt_bboxes,
                                                                      target_gt_idx, fg_mask)

        # Normalize 归一化分数
        # 目标分数不仅考虑到分类概率，还结合了对齐度量和重叠度量，使得分数更加合理和准确
        # 保留仅与正样本锚点相关的对齐指标
        align_metric *= mask_pos
        # 在最后一维（锚点维度）上取最大值，得到每个批次中的最大预测分数。
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # b, max_num_obj
        # 计算锚框iou与正样本掩码的乘积，并在锚点维度上取最大值，得到每个真实框的最大iou值。
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj
        # 计算规范化后的对齐度量，并在第二维度（真实框维度）上取最大值。
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        small_fg_scores_mask = small_fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 80)
        small_target_scores = torch.where(small_fg_scores_mask > 0, target_scores, 0) * norm_align_metric

        return target_labels, target_bboxes, target_scores, small_target_scores, fg_mask.bool(), small_fg_mask.bool(), target_gt_idx

    def get_selected_box_mask(self, gt_bboxes, fg_mask, mask_gt, target_gt_idx, threshold=32):
        # 判断每个真实框是否为小目标（根据宽高或面积定义）
        gt_wh = gt_bboxes[..., 2:] - gt_bboxes[..., :2]  # 计算宽高 (bs, n_max_boxes, 2)
        gt_area = gt_wh[..., 0] * gt_wh[..., 1]  # 计算面积 (bs, n_max_boxes)
        is_small_gt = (gt_area < threshold ** 2) & mask_gt.squeeze(-1).to(torch.bool)  # 小目标 (bs, n_max_boxes)

        # 根据 target_gt_idx 将小目标标签映射到锚点
        small_fg_mask = is_small_gt.gather(1, target_gt_idx)  # 形状 (bs, num_anchors)
        small_fg_mask = small_fg_mask & fg_mask.bool()  # 仅保留正样本锚点

        return small_fg_mask
    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        """Get in_gts mask, (b, max_num_obj, h*w)."""
        # 筛选位于真实框内的锚点, 对于不在真实框内的锚点，其对应的target的分数为0
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes)
        # Get anchor_align metric, (b, max_num_obj, h*w)
        # 计算对齐指标
        # align_metric（对齐指标）为增强后的用真实的分类给这些锚点确定预测分数，使用其对应预测框的分类分数和IoU的指数加权乘积构成， overlaps为预测box和其对应真实box的iou（无效的预测box不更新）
        # align_metric -> [batchsize, max_num_obj, anchors_num], 每个元素值为这个真实标签下这个锚点的预测分数
        # overlaps -> [batchsize, max_num_obj, anchors_num]， 每个元素值为这个真实标签下这个锚框和对应真实框的iou
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        # Get topk_metric mask, (b, max_num_obj, h*w)
        # 对每个真实框，选择 topk 个对齐指标最高的锚点，生成掩码 mask_topk
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """Compute alignment metric given predicted and ground truth bounding boxes."""
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        # 批次索引
        # data: [0, 1,..., self.bs-1] (view)-> [[0], [1],..., [self.bs-1]] (expand)-> [[0, 0, 0,..., 0], [1, 1, 1,..., 1],....]
        # shape: [bs] (view)-> [bs, 1] (expand)-> [bs, n_max_boxes]
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj
        # 真实目标的类别标签
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj

        # Get the scores of each grid for each gt cls
        # 提取对应真实类别的预测分数
        # pd_scores[ind[0], :, ind[1]等价于下面的流程：
        # for i in range(batch_size):
        #     for j in range(max_num_obj):
        #         # 取第i个批次、所有锚点、第gt_labels.squeeze(-1)[i,j]类的预测得分
        #         bbox_scores[i,j] = pd_scores[i, :, gt_labels.squeeze(-1)[i,j]]
        # 就是对于一个图片，预测中，它有最大m个类别，对于每个类别k都有na个锚框，那么这些的锚框的预测分数,可以使用对应真实类别gt_labels中第k个类别的分类l，使用这个l从预测分数pd_scores中取出
        # 就是用真实的分类给这些锚点确定预测分数（每个锚框都根据分类数nc有nc个预测分数）
        # bbox_scores --> [batchsize, max_num_obj, anchors_num] --- 可以表示为对于每张图片最多有max_num_obj个类别，每个类别有anchors_num个锚框，每个值就是这张图片的这个锚框在这个分类下的预测分数
        # mask_gt按照两个条件筛选（要同时满足）：
        #   1.这张图片的真实标签数
        #   2.这张图片的有效锚点(在对应真实框内的锚点）
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w

        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        # bbox_scores --> [batchsize, max_num_obj, anchors_num]
        # 获得有效的预测框
        # pd_bboxes --> [batchsize, anchors_num, [x1, y1, x2, y2]]
        #     (unsqueeze)   -> [batchsize, 1, anchors_num, [x1, y1, x2, y2]]
        #     (expand)      -> [batchsize, max_num_obj, anchors_num, [x1, y1, x2, y2]]
        #     ([mask_gt])   -> [num_valid_anchors, [x1, y1, x2, y2]], num_valid_anchors是有效的锚点数
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        # 获得有效的真实框，相当于上面所有的预测框对应的真实框（这个对应是利用这个预测框的锚点是否在真实框内决定的）
        # gt_bboxes --> [batchsize, max_num_obj, [x1, y1, x2, y2]]
        #     (unsqueeze)   -> [batchsize, max_num_obj, 1, [x1, y1, x2, y2]]
        #     (expand)      -> [batchsize, max_num_obj, anchors_num, [x1, y1, x2, y2]]
        #     ([mask_gt])   -> [num_valid_anchors, [x1, y1, x2, y2]], num_valid_anchors是有效的锚点数
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        # overlaps中只有有效的锚点和其对应真实框的iou被更新
        # 计算预测框与真实框的 CIoU
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)
        # 计算任务对齐指标，alpha，beta控制分类与定位的权重
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """IoU calculation for horizontal bounding boxes."""
        return bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)

    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        """
        Select the top-k candidates based on the given metrics.

        Args:
            metrics (Tensor): A tensor of shape (b, max_num_obj, h*w), where b is the batch size,
                              max_num_obj is the maximum number of objects, and h*w represents the
                              total number of anchor points.
            largest (bool): If True, select the largest values; otherwise, select the smallest values.
            topk_mask (Tensor): An optional boolean tensor of shape (b, max_num_obj, topk), where
                                topk is the number of top candidates to consider. If not provided,
                                the top-k values are automatically computed based on the given metrics.

        Returns:
            (Tensor): A tensor of shape (b, max_num_obj, h*w) containing the selected top-k candidates.
        """
        # (b, max_num_obj, topk)
        # 对于每个锚框对应的真实目标分数，只选择前self.topk个预测分数最高的锚点，同时返回他们的索引
        # topk_metrics -> [batchsize, max_num_obj, topk]
        # topk_idxs -> [batchsize, max_num_obj, topk]
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)
        # 自动生成 topk_mask，标记有效真实目标（避免处理填充的无效目标）
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        # (b, max_num_obj, topk)
        # topk_mask中是图片中不存在的标签，~topk_mask即为那些不存在的标签，masked_fill_能让这些位置的topk_idxs为0
        topk_idxs.masked_fill_(~topk_mask, 0)

        # (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)
        #  统计锚点被选中的次数
        for k in range(self.topk):
            # Expand topk_idxs for each value of k and add 1 at the specified positions
            # 若某个锚点被多次选中（被不同真实目标选中），其计数值会累加
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k : k + 1], ones)
        # count_tensor.scatter_add_(-1, topk_idxs, torch.ones_like(topk_idxs, dtype=torch.int8, device=topk_idxs.device))
        # Filter invalid bboxes
        # 将计数值大于1的位置置为0。
        # 这意味着如果一个锚点被多个真实目标选中（计数值 >1），则最终将其标记为无效（0）。
        count_tensor.masked_fill_(count_tensor > 1, 0)

        return count_tensor.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """
        Compute target labels, target bounding boxes, and target scores for the positive anchor points.

        Args:
            gt_labels (Tensor): Ground truth labels of shape (b, max_num_obj, 1), where b is the
                                batch size and max_num_obj is the maximum number of objects.
            gt_bboxes (Tensor): Ground truth bounding boxes of shape (b, max_num_obj, 4).
            target_gt_idx (Tensor): Indices of the assigned ground truth objects for positive
                                    anchor points, with shape (b, h*w), where h*w is the total
                                    number of anchor points.
            fg_mask (Tensor): A boolean tensor of shape (b, h*w) indicating the positive
                              (foreground) anchor points.

        Returns:
            (Tuple[Tensor, Tensor, Tensor]): A tuple containing the following tensors:
                - target_labels (Tensor): Shape (b, h*w), containing the target labels for
                                          positive anchor points.
                - target_bboxes (Tensor): Shape (b, h*w, 4), containing the target bounding boxes
                                          for positive anchor points.
                - target_scores (Tensor): Shape (b, h*w, num_classes), containing the target scores
                                          for positive anchor points, where num_classes is the number
                                          of object classes.
        """
        # Assigned target labels, (b, 1)
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w)
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)

        # Assigned target boxes, (b, max_num_obj, 4) -> (b, h*w, 4)
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]

        # Assigned target scores
        target_labels.clamp_(0)

        # 10x faster than F.one_hot()
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
            dtype=torch.int64,
            device=target_labels.device,
        )  # (b, h*w, 80)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)

        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 80)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        return target_labels, target_bboxes, target_scores

    @staticmethod
    def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
        """
        Select positive anchor centers within ground truth bounding boxes.

        Args:
            xy_centers (torch.Tensor): Anchor center coordinates, shape (h*w, 2).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes, shape (b, n_boxes, 4).
            eps (float, optional): Small value for numerical stability. Defaults to 1e-9.

        Returns:
            (torch.Tensor): Boolean mask of positive anchors, shape (b, n_boxes, h*w).

        Note:
            b: batch size, n_boxes: number of ground truth boxes, h: height, w: width.
            Bounding box format: [x_min, y_min, x_max, y_max].
        """
        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape
        # lt， rb分别为所有的真实框左上角，右下角点的坐标
        # gt_bboxes --> [batchsize, max_labels_num, [x1,y1,x2,y2]] -> [batchsize*max_labels_num, 1, [x1,y1,x2,y2]]
        # lt --> [batchsize*max_labels_num, 1, [x1,y1]]
        # rb --> [batchsize*max_labels_num, 1, [x2,y2]]
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
        # 这里求解出真实框点和所有锚点的距离
        # xy_centers --> [anchors_num, [x,y]] xy_centers[None] --> [1, anchors_num, [x,y]]
        # xy_centers[None] - lt -> [1, anchors_num,[x,y]] - [batchsize*max_labels_num, 1, [x1,y1]
        #                       -> [batchsize*max_labels_num, anchors_num, [x,y]] - [batchsize*max_labels_num, anchors_num, [x1,y1]
        #                       -> [batchsize*max_labels_num, anchors_num, [(x - x1),(y - y1)]]
        #       (torch.cat)     -> [batchsize*max_labels_num, anchors_num, [(x - x1),(y - y1),(x2 - x),(y2 - y)]]
        #       (view)          -> [batchsize, max_labels_num, anchors_num, [(x - x1),(y - y1),(x2 - x),(y2 - y)]] <-- bbox_deltas
        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
        # return (bbox_deltas.min(3)[0] > eps).to(gt_bboxes.dtype)
        #       (amin)          -> [batchsize, max_labels_num, anchors_num]
        # 每个元素代表这个图片在这个真实标签下，其表示真实框位置的两个点的四个坐标值和其中一个anchor点的xy差值的最小值，同时这个最小值要大于eps(最小值要是正数）
        # True表示这个锚点在真实的标定框内，False为不在
        return bbox_deltas.amin(3).gt_(eps)

    @staticmethod
    def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
        """
        Select anchor boxes with highest IoU when assigned to multiple ground truths.

        Args:
            mask_pos (torch.Tensor): Positive mask, shape (b, n_max_boxes, h*w).
            overlaps (torch.Tensor): IoU overlaps, shape (b, n_max_boxes, h*w).
            n_max_boxes (int): Maximum number of ground truth boxes.

        Returns:
            target_gt_idx (torch.Tensor): Indices of assigned ground truths, shape (b, h*w).
            fg_mask (torch.Tensor): Foreground mask, shape (b, h*w).
            mask_pos (torch.Tensor): Updated positive mask, shape (b, n_max_boxes, h*w).

        Note:
            b: batch size, h: height, w: width.
        """
        # Convert (b, n_max_boxes, h*w) -> (b, h*w)
        # 在 n_max_boxes 维度上求和，得到形状为 (b, h*w) 的 fg_mask。这表示每个锚点同时属于多少个真实分类
        fg_mask = mask_pos.sum(-2)
        # 检查是否存在一个锚点被多个真实框分配
        if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
            # 创建多重匹配的掩码，表示那些被多重分配的锚点
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)  # (b, n_max_boxes, h*w)
            # 对于每个锚点，找到与之具有最大IoU的真实框的索引
            max_overlaps_idx = overlaps.argmax(1)  # (b, h*w)
            # 创建仅保留最大IoU匹配的掩码
            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
            # 对于多重匹配的锚点，仅保留与其具有最大IoU的真实框匹配，其余的匹配置零。
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()  # (b, n_max_boxes, h*w)
            fg_mask = mask_pos.sum(-2)
        # Find each grid serve which gt(index)
        # 确定每个锚点最终分配的真实框索引
        target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
        return target_gt_idx, fg_mask, mask_pos


class RotatedTaskAlignedAssigner(TaskAlignedAssigner):
    """Assigns ground-truth objects to rotated bounding boxes using a task-aligned metric."""

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """IoU calculation for rotated bounding boxes."""
        return probiou(gt_bboxes, pd_bboxes).squeeze(-1).clamp_(0)

    @staticmethod
    def select_candidates_in_gts(xy_centers, gt_bboxes):
        """
        Select the positive anchor center in gt for rotated bounding boxes.

        Args:
            xy_centers (Tensor): shape(h*w, 2)
            gt_bboxes (Tensor): shape(b, n_boxes, 5)

        Returns:
            (Tensor): shape(b, n_boxes, h*w)
        """
        # (b, n_boxes, 5) --> (b, n_boxes, 4, 2)
        corners = xywhr2xyxyxyxy(gt_bboxes)
        # (b, n_boxes, 1, 2)
        a, b, _, d = corners.split(1, dim=-2)
        ab = b - a
        ad = d - a

        # (b, n_boxes, h*w, 2)
        ap = xy_centers - a
        norm_ab = (ab * ab).sum(dim=-1)
        norm_ad = (ad * ad).sum(dim=-1)
        ap_dot_ab = (ap * ab).sum(dim=-1)
        ap_dot_ad = (ap * ad).sum(dim=-1)
        return (ap_dot_ab >= 0) & (ap_dot_ab <= norm_ab) & (ap_dot_ad >= 0) & (ap_dot_ad <= norm_ad)  # is_in_box


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        h, w = feats[i].shape[2:] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1]))
        # 以1为步长，偏置為grid_cell_offset，从0到end生成一个等差数列
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset# shift y
        # 将sx，sy排列组合
        sy, sx = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_10 else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        # 生成等大小矩阵，元素全为stride
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    # lt, rb 分别为对应anchor点到预测框左上角和右下角点的x，y的距离，故由此可知是在回归距离
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)  # dist (lt, rb)


def dist2rbox(pred_dist, pred_angle, anchor_points, dim=-1):
    """
    Decode predicted rotated bounding box coordinates from anchor points and distribution.

    Args:
        pred_dist (torch.Tensor): Predicted rotated distance, shape (bs, h*w, 4).
        pred_angle (torch.Tensor): Predicted angle, shape (bs, h*w, 1).
        anchor_points (torch.Tensor): Anchor points, shape (h*w, 2).
        dim (int, optional): Dimension along which to split. Defaults to -1.

    Returns:
        (torch.Tensor): Predicted rotated bounding boxes, shape (bs, h*w, 4).
    """
    lt, rb = pred_dist.split(2, dim=dim)
    cos, sin = torch.cos(pred_angle), torch.sin(pred_angle)
    # (bs, h*w, 1)
    xf, yf = ((rb - lt) / 2).split(1, dim=dim)
    x, y = xf * cos - yf * sin, xf * sin + yf * cos
    xy = torch.cat([x, y], dim=dim) + anchor_points
    return torch.cat([xy, lt + rb], dim=dim)
