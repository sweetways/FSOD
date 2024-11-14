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

        # 初始化Teacher模型的backbone，并复制Student模型的参数
        self.teacher_backbone = build_backbone(cfg)
                # 加载原始检查点
        checkpoint = torch.load(cfg.MODEL.TEACHER_WEIGHT)
        state_dict = checkpoint['model']  # 根据您的检查点结构获取 state_dict
        self.teacher_features_cache = None
        self.cache_update_interval = cfg.TRAIN.CACHE_UPDATE_INTERVAL  # 每隔一定步长更新缓存

        step_count = 0
        # 创建一个新的 state_dict，用于存储修改后的参数名称
        new_state_dict = {}
        for k, v in state_dict.items():
            # 如果参数名称包含 'backbone.student_' 前缀，去掉该前缀
            if k.startswith('backbone.'):
                new_key = k.replace('backbone.', '')
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v

        self.teacher_backbone.load_state_dict(new_state_dict, strict=False)

        # 冻结Teacher模型的参数
        for param in self.teacher_backbone.parameters():
            param.requires_grad = False

    # 定义对比损失函数
    def contrastive_loss(self, student_features, teacher_features, temperature=0.5):
        student_features = F.normalize(student_features, dim=-1)
        teacher_features = F.normalize(teacher_features, dim=-1)
        logits = torch.matmul(student_features, teacher_features.T) / temperature
        labels = torch.arange(logits.size(0)).long().to(logits.device)
        loss = F.cross_entropy(logits, labels)
        return loss

    def forward(self, batched_inputs):
        # 调用父类的方法，获取Student模型的输出
        if not self.training:
            return super().forward(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        # Student模型的前向传播
        proposals, proposal_losses = self.proposal_generator(images, features, None)
        _, detector_losses = self.roi_heads(images, features, proposals, None)

        # 提取Student模型的特征
        student_features = features[next(iter(features))]

        # Teacher模型的前向传播
        if self.training:
            # 判断是否需要更新教师特征缓存
            if self.step_count % self.cache_update_interval == 0:
                with torch.no_grad():
                    teacher_features = self.teacher_backbone(images.tensor)
                    self.teacher_features_cache = teacher_features[next(iter(teacher_features))]


        # 计算对比损失
        contrastive_loss = self.contrastive_loss(student_features, teacher_features)
        self.step_count += 1
        # 合并损失
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        losses['contrastive_loss'] = contrastive_loss
        return losses