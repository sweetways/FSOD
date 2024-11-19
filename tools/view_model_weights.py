import torch

def load_model_weights(file_path):
    """
    加载模型权重文件并打印参数名称和形状
    :param file_path: 模型权重文件的路径
    """
    # 加载模型权重文件
    checkpoint = torch.load(file_path, map_location=torch.device('cuda'))

    # 检查点可能包含多个键，例如 'model'、'optimizer' 等
    # 我们只关心 'model' 键中的参数
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # 打印参数名称和形状
    for name, param in state_dict.items():
        print(f"参数名称: {name}, 形状: {param.shape}")

if __name__ == "__main__":
    # 模型权重文件的路径
    model_weights_path = "/root/FSOD/checkpoints/voc/prior/ETFRes_pre1_10shot_lr20_adj7.0_rfs1.0_t1/model_clean_student.pth"

    # 加载并打印模型权重
    load_model_weights(model_weights_path)