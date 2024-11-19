# rcnn.py

import torch
import torch.nn.functional as F

# 引入必要的模块和类
from .build import META_ARCH_REGISTRY
from fsdet.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.modeling.backbone import build_backbone

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
    def contrastive_loss(self, student_features, teacher_features, temperature=0.5):
        # 提取特定的特征层，例如最后一层特征
        student_features = student_features[next(iter(student_features))]
        teacher_features = teacher_features[next(iter(teacher_features))]
        if torch.isnan(student_features).any() or torch.isinf(student_features).any():
            raise ValueError("Student features contain NaN or Inf values")
        if torch.isnan(teacher_features).any() or torch.isinf(teacher_features).any():
            raise ValueError("Teacher features contain NaN or Inf values")
        # 对特征进行归一化处理
        student_features = F.normalize(student_features, dim=-1)
        teacher_features = F.normalize(teacher_features, dim=-1)

        # 调整特征层的形状使其匹配
        if student_features.shape != teacher_features.shape:
            teacher_features = F.interpolate(teacher_features, size=student_features.shape[2:], mode='bilinear', align_corners=False)

        # 计算对比损失
        student_features_flat = student_features.view(student_features.size(0), -1)
        teacher_features_flat = teacher_features.view(teacher_features.size(0), -1)
        logits = torch.matmul(student_features_flat, teacher_features_flat.T) / temperature
        labels = torch.arange(logits.size(0)).long().to(logits.device)
        loss = F.cross_entropy(logits, labels)
        return loss

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

        
        # 计算对比损失
        contrastive_loss = self.contrastive_loss(student_features, teacher_features, 1)
        self.step_count += 1

        # 从配置中获取对比损失权重
        contrastive_loss_weight = self.cfg.MODEL.CONTRASTIVE_WEIGHT
        contrastive_loss = contrastive_loss * contrastive_loss_weight
        # 合并损失
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        losses['contrastive_loss'] = contrastive_loss

        return losses