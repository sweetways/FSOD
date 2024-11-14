import time
import os
import torch
from fsdet.engine import DefaultTrainer
from fsdet.modeling.meta_arch.build import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.transforms import Resize


class DistillTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if cfg.MODEL.IS_DISTILL:
            # Initialize teacher model and load features if they exist
            self.teacher_model = self.build_teacher(cfg)
            self.teacher_feature_path = "teacher_features.pth"

            if os.path.isfile(self.teacher_feature_path):
                self.teacher_features = self.load_teacher_features()
            else:
                print("Teacher feature file not found. Extracting features...")
                self.teacher_features = self.extract_and_save_teacher_features(self.data_loader, self.teacher_model, self.device, self.teacher_feature_path)
        else:
            self.teacher_model = None
            self.teacher_features = None

    def build_teacher(self, cfg):
        """Construct and load the teacher model with frozen backbone."""
        cfg_teacher = cfg.clone()
        cfg_teacher.defrost()
        cfg_teacher.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"
        cfg_teacher.MODEL.BACKBONE.FREEZE = True
        cfg_teacher.MODEL.BACKBONE.FREEZE_AT = 0
        cfg_teacher.freeze()
        
        teacher_model = build_model(cfg_teacher)
        teacher_weight_path = cfg.MODEL.TEACHER_WEIGHT
        checkpoint = torch.load(teacher_weight_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint["model"], strict=False)
        teacher_model.to(self.device).eval()
        
        return teacher_model

    def load_teacher_features(self):
        """Load saved teacher features for reuse."""
        return torch.load(self.teacher_feature_path, map_location=self.device)

    def extract_and_save_teacher_features(self, data_loader, teacher_model, device, save_path="teacher_features.pth"):
        return 

    def run_step(self):
        """单步训练逻辑，包含对比学习损失计算。"""
        assert self.model.training, "[Trainer] model was not在训练模式中!"
        start = time.perf_counter()

        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        images = data[0]["image"].to(self.device)
        student_features = self.model.backbone(images)  # 获取学生模型的特征
        
        # 计算分类损失
        loss_dict = {
            "classification_loss": torch.nn.functional.cross_entropy(student_features["res4"], data[0]["label"])
        }

        # 如果有教师特征，则计算对比损失
        if self.cfg.MODEL.USE_CONTRASTIVE and self.teacher_features:
            batch_idx = self._data_loader_iter._idx % len(self.teacher_features)
            teacher_features = self.teacher_features[batch_idx].to(self.device)
            
            # 计算对比损失（假设 `res4` 是目标特征层）
            contrastive_loss = torch.nn.functional.mse_loss(student_features["res4"], teacher_features)
            loss_dict["contrastive_loss"] = self.cfg.MODEL.CONTRASTIVE_WEIGHT * contrastive_loss

        # 计算总损失
        losses = sum(loss_dict.values())
        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

        # 记录损失
        self._write_metrics(loss_dict, data_time)
