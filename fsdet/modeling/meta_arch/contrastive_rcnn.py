# rcnn.py

import torch
import torch.nn.functional as F
import torch.nn as nn

# 引入必要的模块和类
from .build import META_ARCH_REGISTRY
from fsdet.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.modeling.backbone import build_backbone
from detectron2.layers import ROIAlign
from detectron2.structures import pairwise_iou, Instances, Boxes
from detectron2.utils.events import get_event_storage
from detectron2.utils.visualizer import Visualizer
import matplotlib.pyplot as plt
import numpy as np
from fsdet.modeling.attention.self_attention import selfAttention


@META_ARCH_REGISTRY.register()
class ContrastiveGeneralizedRCNN(GeneralizedRCNN):
    def __init__(self, cfg):
        super().__init__(cfg)

        # 初始化Teacher模型的路径，而不是直接创建它
        self.teacher_backbone_path = cfg.MODEL.TEACHER_WEIGHT
        self.teacher_features_cache = None
        self.cache_update_interval = cfg.TRAIN.CACHE_UPDATE_INTERVAL  # 每隔一定步长更新缓存
        self.cfg = cfg
        self.step_count = 0
        self.temperature = cfg.MODEL.TEMPERATURE

        # self.self_attention_blocks = nn.ModuleDict({
        #     "p3": selfAttention(embed_dim=256, num_heads=8),  # 256是FPN p3的通道数
        #     "p4": selfAttention(embed_dim=256, num_heads=8)   # 256是FPN p4的通道数
        # })


    def _initialize_teacher_backbone(self):
        teacher_backbone = build_backbone(self.cfg)
        teacher_backbone.to(self.device)
        # 加载原始检查点
        checkpoint = torch.load(self.teacher_backbone_path)
        # 创建一个新的 state_dict，用于存储修改后的参数名称
        state_dict = checkpoint['model']  # 根据您的检查点结构获取 state_dict
        new_state_dict = {}
        for k, v in state_dict.items():
            # 如果参数名称包含 'backbone.' 前缀，去掉该前缀
            if k.startswith('backbone.'):
                new_key = k.replace('backbone.', '')
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v

        teacher_backbone.load_state_dict(new_state_dict, strict=False)

        # 冻结Teacher模型的参数
        for param in teacher_backbone.parameters():
            param.requires_grad = False

        return teacher_backbone
  

    def contrastive_loss(self, student_features, teacher_features, positive_samples,
                        negative_samples, proposals, batched_inputs, temperature=0.5):
        # 定义 RoI Align 层
        feature_key = "p4"  # 根据你的模型选择合适的特征层
        roi_align = ROIAlign(
            output_size=(7, 7),  # 输出尺寸，这里将特征池化为 1x1
            spatial_scale=1.0 / self.backbone.output_shape()[feature_key].stride,  # 特征图与原图的尺寸比例
            sampling_ratio=0,
            aligned=True
        )

        # 准备 RoI 盒子
        def get_rois(samples):
            rois = []
            for idx in samples:
                batch_idx = idx // self.num_proposals_per_image
                feature_idx = idx % self.num_proposals_per_image
                # 获取对应的 proposal_box
                proposal_box = proposals[batch_idx].proposal_boxes[feature_idx]
                # 构建 RoI：[batch_idx, x1, y1, x2, y2]
                box = proposal_box.tensor  # shape: [1, 4]
                roi = torch.cat([torch.tensor([[batch_idx]], device=box.device), box], dim=1)  # [1, 5]
                rois.append(roi)
            if rois:
                return torch.cat(rois, dim=0)  # [N, 5]
            else:
                return torch.zeros((0, 5), device=student_features[feature_key].device)

        positive_rois = get_rois(positive_samples)
        negative_rois = get_rois(negative_samples)

        # 提取特征
        student_feat_map = student_features[feature_key]
        teacher_feat_map = teacher_features[feature_key]

        if positive_rois.size(0) == 0 or negative_rois.size(0) == 0:
            # 如果没有正样本或负样本，返回零损失
            return torch.tensor(0.0, device=student_feat_map.device)

        positive_student_features = roi_align(student_feat_map, positive_rois)
        negative_student_features = roi_align(student_feat_map, negative_rois)

        positive_teacher_features = roi_align(teacher_feat_map, positive_rois)
        negative_teacher_features = roi_align(teacher_feat_map, negative_rois)

        # 展平特征
        positive_student_features = positive_student_features.view(positive_student_features.size(0), -1)
        negative_student_features = negative_student_features.view(negative_student_features.size(0), -1)
        positive_teacher_features = positive_teacher_features.view(positive_teacher_features.size(0), -1)
        negative_teacher_features = negative_teacher_features.view(negative_teacher_features.size(0), -1)

        # 对特征进行归一化
        positive_student_features = F.normalize(positive_student_features, dim=1)
        negative_student_features = F.normalize(negative_student_features, dim=1)
        positive_teacher_features = F.normalize(positive_teacher_features, dim=1)
        negative_teacher_features = F.normalize(negative_teacher_features, dim=1)

        max_pos_sim = torch.max(positive_student_features * positive_teacher_features, dim=1, keepdim=True)[0]
        pos_sim = torch.exp((torch.sum(positive_student_features * positive_teacher_features, dim=1) - max_pos_sim) / temperature)

        max_neg_sim = torch.max(positive_student_features @ negative_teacher_features.t(), dim=1, keepdim=True)[0]
        neg_sim = torch.exp((positive_student_features @ negative_teacher_features.t() - max_neg_sim) / temperature)
        neg_sim = neg_sim.sum(dim=1)

        # 计算 InfoNCE 损失
        loss = -torch.log(pos_sim / (pos_sim + neg_sim))
        return loss.mean()

    def forward(self, batched_inputs):
        if not self.training:
            return super().forward(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        # 提取目标ground truth
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        # Student 模型的前向传播
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)

        # 提取 Student 模型的特征
        student_features = features


        # Teacher 模型的前向传播
        if self.training:
            if not hasattr(self, 'teacher_backbone'):
                self.teacher_backbone = self._initialize_teacher_backbone()  # 初始化教师模型
            with torch.no_grad():
                teacher_features = self.teacher_backbone(images.tensor)  # 使用当前批次的图像计算特征
                self.teacher_features_cache = teacher_features  # 更新教师特征缓存
        # 提取 Student 模型的特征
        # 提取用于对比学习的特定层次的特征
        # fpn_keys = ["p3", "p4"]
        # student_features = {key: features[key] for key in fpn_keys}
        # teacher_features = {key: self.teacher_features_cache[key] for key in fpn_keys if key in self.teacher_features_cache}

        positive_samples, negative_samples = self.assign_positive_negative_samples(proposals, 
                                                                                   gt_instances)
        # 计算对比损失
        contrastive_loss = self.contrastive_loss(student_features, teacher_features, positive_samples, negative_samples, proposals, batched_inputs, self.temperature)
        self.step_count += 1

        # 从配置中获取对比损失权重
        contrastive_loss_weight = self.cfg.MODEL.CONTRASTIVE_WEIGHT
        contrastive_loss = contrastive_loss * contrastive_loss_weight
        # 合并损失
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        losses['contrastive_loss'] = contrastive_loss

        # self.visualize_results(images, proposals, gt_instances[0] if gt_instances else None)
        return losses
    
    def compute_iou(self, boxes1, boxes2):
        """
        使用 Detectron2 提供的 pairwise_iou 函数计算 IoU
        """
        return pairwise_iou(boxes1, boxes2)

    def assign_positive_negative_samples(self, proposals, gt_instances, iou_threshold_pos=0.5, iou_threshold_neg=0.3, some_threshold=3):
        """
        使用 Detectron2 工具划分正负样本
        """
        # 提取 proposals 和 ground truth boxes
        self.num_proposals_per_image = proposals[0].proposal_boxes.tensor.shape[0]
        proposal_boxes = torch.cat([p.proposal_boxes.tensor for p in proposals], dim=0)
        gt_boxes = torch.cat([gt.gt_boxes.tensor for gt in gt_instances], dim=0)

        proposal_boxes = Boxes(proposal_boxes)
        gt_boxes = Boxes(gt_boxes)
        # 计算 IoU
        iou_matrix = pairwise_iou(proposal_boxes, gt_boxes)

        positive_samples = []
        negative_samples = []

        # 逐个proposal，找到最大IoU和相应的GT
        proposal_idx_offset = 0  # 用于计算proposal属于哪张图片
        gt_idx_offset = 0  # 用于计算gt属于哪张图片
        for img_idx, (proposal, gt) in enumerate(zip(proposals, gt_instances)):
            # 获取当前图像的proposals和gt的数量
            num_proposals = len(proposal.proposal_boxes)
            num_gt_boxes = len(gt.gt_boxes)

            # 从iou_matrix中获取该图像的IoU矩阵
            iou_sub_matrix = iou_matrix[proposal_idx_offset:proposal_idx_offset+num_proposals, 
                                        gt_idx_offset:gt_idx_offset+num_gt_boxes]
            
            for i, iou in enumerate(iou_sub_matrix):
                max_iou = iou.max()
                gt_index = iou.argmax()
                
                score = proposal.objectness_logits[i].item()  
                if max_iou >= iou_threshold_pos and score > some_threshold:
                    positive_samples.append(i + proposal_idx_offset)  # 记录全局索引
                elif max_iou < iou_threshold_neg:
                    negative_samples.append(i + proposal_idx_offset)  # 记录全局索引

            # 更新proposal和gt的偏移量
            proposal_idx_offset += num_proposals
            gt_idx_offset += num_gt_boxes

        return positive_samples, negative_samples
    
    def visualize_results(self, images, proposals, targets=None):
        """
        使用 Detectron2 的可视化工具输出检测结果
        """
        img = images[0].cpu().numpy().transpose(1, 2, 0)
        v = Visualizer(img)

        # 绘制 proposals
        if proposals:
            proposal_boxes = proposals.proposal_boxes.tensor.cpu().numpy()
            v = v.overlay_instances(boxes=proposal_boxes)

        # 绘制 Ground Truth
        if targets:
            gt_boxes = targets.gt_boxes.tensor.cpu().numpy()
            v = v.overlay_instances(boxes=gt_boxes, alpha=0.5, color="green")

        result_img = v.get_output().get_image()

        # 使用 matplotlib 可视化
        plt.figure(figsize=(12, 8))
        plt.imshow(result_img)
        plt.axis('off')
        plt.show()

    def plot_contrastive_loss(self):
        """
        可视化对比学习的损失变化
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.contrastive_losses, label='Contrastive Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Contrastive Loss During Training')
        plt.legend()
        plt.grid(True)
        plt.show()