# rcnn.py

import torch
import torch.nn.functional as F
import torch.nn as nn

# 引入必要的模块和类
from .build import META_ARCH_REGISTRY
from fsdet.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.modeling.backbone import build_backbone

from detectron2.structures import pairwise_iou, Instances
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

        self.self_attention_blocks = nn.ModuleDict({
            "p3": selfAttention(embed_dim=256, num_heads=8),  # 256是FPN p3的通道数
            "p4": selfAttention(embed_dim=256, num_heads=8)   # 256是FPN p4的通道数
        })


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
    # 定义对比损失函数
    def contrastive_loss(self, student_features, teacher_features, positive_samples,
                         negative_samples, temperature=0.5):
        # 提取特定的特征层，例如最后一层特征
        student_features = student_features[next(iter(student_features))]
        teacher_features = teacher_features[next(iter(teacher_features))]

        # 对特征进行归一化处理
        student_features = F.normalize(student_features, dim=-1)
        teacher_features = F.normalize(teacher_features, dim=-1)

        # 调整特征层的形状使其匹配
        if student_features.shape != teacher_features.shape:
            teacher_features = F.interpolate(teacher_features, size=student_features.shape[2:], mode='bilinear', align_corners=False)

        # 正样本对比
        positive_student_features = student_features[positive_samples]
        positive_teacher_features = teacher_features[positive_samples]
        pos_sim = torch.exp(torch.sum(positive_student_features * positive_teacher_features, dim=-1) / temperature)

        # 负样本对比
        negative_student_features = student_features[negative_samples]
        negative_teacher_features = teacher_features[negative_samples]
        neg_sim = torch.exp(torch.sum(negative_student_features * negative_teacher_features, dim=-1) / temperature)

        # 计算 InfoNCE 损失
        loss = -torch.log(pos_sim / (pos_sim + neg_sim.sum(dim=0, keepdim=True)))
        return loss.mean()

    def forward(self, batched_inputs):
        if not self.training:
            return super().forward(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        # 提取目标实例
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        # Student 模型的前向传播
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)

        # 提取 Student 模型的特征
        student_features = features

        # Teacher 模型的前向传播
        if self.training:
            # 判断是否需要更新教师特征缓存
            if self.step_count % self.cache_update_interval == 0:
                # 只在当需要时初始化 Teacher 模型
                teacher_backbone = self._initialize_teacher_backbone()
                with torch.no_grad():
                    teacher_features = teacher_backbone(images.tensor.to(self.device))
                    self.teacher_features_cache = teacher_features
            else:
                teacher_features = self.teacher_features_cache
        # 提取 Student 模型的特征
        # 提取用于对比学习的特定层次的特征
        fpn_keys = ["p3", "p4"]
        student_features = {key: features[key] for key in fpn_keys}
        teacher_features = {key: self.teacher_features_cache[key] for key in fpn_keys if key in self.teacher_features_cache}

        positive_samples, negative_samples = self.assign_positive_negative_samples(proposals, 
                                                                                   Instances(gt_boxes=[x.gt_boxes for x in gt_instances], 
                                                                                             gt_classes=[x.gt_classes for x in gt_instances]))
        # 计算对比损失
        contrastive_loss = self.contrastive_loss(student_features, teacher_features, positive_samples, negative_samples, 1)
        self.step_count += 1

        # 从配置中获取对比损失权重
        contrastive_loss_weight = self.cfg.MODEL.CONTRASTIVE_WEIGHT
        contrastive_loss = contrastive_loss * contrastive_loss_weight
        # 合并损失
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        losses['contrastive_loss'] = contrastive_loss

        self.visualize_results(images, proposals, gt_instances[0] if gt_instances else None)
        return losses
    
    def compute_iou(self, boxes1, boxes2):
        """
        使用 Detectron2 提供的 pairwise_iou 函数计算 IoU
        """
        return pairwise_iou(boxes1, boxes2)

    def assign_positive_negative_samples(self, proposals, gt_instances, iou_threshold_pos=0.5, iou_threshold_neg=0.3):
        """
        使用 Detectron2 工具划分正负样本
        """
        # 提取 proposals 和 ground truth boxes
        proposal_boxes = proposals.proposal_boxes.tensor
        gt_boxes = gt_instances.gt_boxes.tensor

        # 计算 IoU
        iou_matrix = pairwise_iou(proposal_boxes, gt_boxes)

        positive_samples = []
        negative_samples = []

        for i, iou in enumerate(iou_matrix):
            max_iou = iou.max()
            gt_index = iou.argmax()
            if max_iou >= iou_threshold_pos and proposals.gt_classes[i] == gt_instances.gt_classes[gt_index]:
                positive_samples.append(i)
            elif max_iou < iou_threshold_neg:
                negative_samples.append(i)

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
