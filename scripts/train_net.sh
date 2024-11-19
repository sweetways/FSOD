split=3
shot=10


set -x

# CUDA_VISIBLE_DEVICES=${gpu} python3 -m tools.test_net --num-gpus 2 --eval-only \
#     --config-file configs/PascalVOC-detection/split${split}/faster_rcnn_R_101_FPN_ETF_pre${split}_${shot}shot.yaml 

nohup python3 -m tools.train_net --num-gpus 2  \
    --config-file configs/PascalVOC-detection/split${split}/faster_rcnn_R_101_FPN_ETF_distill${split}_${shot}shot.yaml > logs/firstDistill${split}_${shot}VOC_Con.log 2>&1 &