import argparse
import os
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument(
        "--src", type=str, default="../checkpoints/voc/prior/", help="Path to the main checkpoint"
    )
    args = parser.parse_args()
    return args

def reset_ckpt(ckpt):
    """
    Resets certain components of the checkpoint for reinitialization of the student model.
    This includes removing training-specific parameters like the optimizer state, scheduler state, etc.
    """
    # Remove optimizer and scheduler from the checkpoint to reset for a fresh start
    if "scheduler" in ckpt:
        del ckpt["scheduler"]
    if "optimizer" in ckpt:
        del ckpt["optimizer"]
    if "iteration" in ckpt:
        ckpt["iteration"] = 0

    # Copy backbone and box predictor weights for student model initialization
    for key in [k for k in ckpt['model'] if 'box_' in k or 'backbone' in k]:
        new_weight = ckpt['model'][key].clone()
        key_comp = key.split('.')
        # Add 'student_' prefix to distinguish student weights
        new_key = '.'.join(key_comp[:1] + ['student_' + key_comp[1]] + key_comp[2:])
        ckpt['model'][new_key] = new_weight

    print("Checkpoint reset: optimizer and scheduler removed, student model weights created.")


if __name__ == "__main__":
    args = parse_args()

    # Iterate over experiments in the source directory
    for exp in os.listdir(args.src):
        # Construct source checkpoint path and destination checkpoint path
        src_ckpt = os.path.join(args.src, exp, 'model_final.pth')
        dst_ckpt = src_ckpt.replace('model_final', 'model_clean_student')

        # Check if the cleaned student checkpoint already exists
        if os.path.isfile(dst_ckpt):
            print(f'The cleaned model for distillation stage already exists for Exp {exp}')
            continue
        elif not os.path.isfile(src_ckpt):
            print(f'The final model is not found, please check {exp}')
            continue

        # Load the checkpoint from the teacher model
        ckpt = torch.load(src_ckpt)
        # Reset the checkpoint to prepare for student model initialization
        reset_ckpt(ckpt)

        # Save the cleaned checkpoint for the student model
        torch.save(ckpt, dst_ckpt)
        print(f'The final model has been cleaned for {exp}')
