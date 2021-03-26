import os
import random
 
root_dir = 'datasets/aquariumvoc/'
 
## 0.7train 0.1val 0.2test
trainval_percent = 0.8
train_percent = 0.7
xmlfilepath = root_dir + 'Annotations'
txtsavepath = root_dir + 'ImageSets/Main'
total_xml = os.listdir(xmlfilepath)
num = len(total_xml)  # 100
list = range(num)
tv = int(num * trainval_percent)  # 80
tr = int(tv * train_percent)  # 80*0.7=56
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)
ftrainval = open(root_dir + 'ImageSets/Main/trainval.txt', 'w+')
ftest = open(root_dir + 'ImageSets/Main/test.txt', 'w+')
ftrain = open(root_dir + 'ImageSets/Main/train.txt', 'w+')
fval = open(root_dir + 'ImageSets/Main/val.txt', 'w+')
for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)
ftrainval.close()
ftrain.close()
fval.close()











# python -m tools.train_net --num-gpus 1 --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_base1.yaml

# python3 -m tools.run_experiments --num-gpus 1 --shots 1--seeds 0 30 --split 3


# http://dl.yf.io/fs-det/models/voc/split3/base_model/

# wget -r -np -R "index.html*" http://dl.yf.io/fs-det/models/voc/split3/base_model/
# wget -r -np -R "index.html*" http://dl.yf.io/fs-det/models/voc/split3/FRCN%2Bft-full_1shot/
# wget -r -np -R "index.html*" http://dl.yf.io/fs-det/models/voc/split3/tfa_cos_1shot/
# wget -r -np -R "index.html*" http://dl.yf.io/fs-det/models/voc/split3/tfa_fc_1shot/


# wget -r -np -R "index.html*" http://dl.yf.io/fs-det/datasets/cocosplit/

# python tools/test_net.py --num-gpus 1 \
#         --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_1shot.yaml \
#         --eval-only


# python tools/run_experiments.py --num-gpus 1 \
#         --shots 1 --seeds 0 10--split 3


# meta data evaluator
# dataset evaluator



# python tools/ckpt_surgery.py \
#         --src1 dl.yf.io/fs-det/models/voc/split3/base_model/model_final.pth \
#         --method randinit \
#         --save-dir checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base3

# python tools/ckpt_surgery.py \
#         --src1 dl.yf.io/fs-det/models/voc/split3/base_model/model_final.pth \
#         --method remove \
#         --save-dir checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_all3


# python tools/train_net.py --num-gpus 1 \
#         --config-file configs/PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_all3_1shot.yaml \
#         --eval-during-train
#         aquarium_voc_trainval_all1_1shot
#         voc_2007_trainval_all3_1shot
        




# python tools/train_net.py --num-gpus 1 --config-file configs/PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_all3_1shot.yaml MODEL.WEIGHTS checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_all3/model_reset_surgery.pth