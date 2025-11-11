import argparse
from tqdm import tqdm
import os
from utils.data3d import NYU
from torch.utils.data import DataLoader
import torch
from utils.cuda import get_device
from utils.metrics import MIoU, CompletionIoU
from model.SDNET import SDNET
import torch.nn.functional as F
import numpy as np



# default settings
GPU =0

# Dataloader settings
BATCH_SIZE = 1
WORKERS = 2
DATASET = "NYU"
PREPROC_PATH = '/your_SSC_datasets_path/SSC_datasets/NYU'

# Model settings
WEIGHTS = "your weight path"
BATCH_NORM = True
INPUT_TYPE = "depth+tsdf"

def parse_arguments():
    global GPU, BATCH_SIZE, WORKERS, DATASET, PREPROC_PATH, \
        WEIGHTS, BATCH_NORM, INPUT_TYPE

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Target dataset", type=str,default=DATASET, choices=['NYU', 'NYUCAD'])
    parser.add_argument("--batch_size", help="Training batch size. Default: " + str(BATCH_SIZE),
                        type=int, default=BATCH_SIZE, required=False)
    parser.add_argument("--weights", help="Pretraind weights. ", default=WEIGHTS,type=str)
    parser.add_argument("--workers", help="Concurrent threads. Default " + str(WORKERS),
                        type=int, default=WORKERS, required=False)
    parser.add_argument("--gpu", help="GPU device. Default " + str(GPU),
                        type=int, default=GPU, required=False)
    parser.add_argument("--input_type", help="Network input type. Default " + INPUT_TYPE,
                        type=str, default=INPUT_TYPE, required=False,
                        choices=['rgb+normals', 'rgb+depth', 'depth']
                        )
    parser.add_argument("--bn", help="Apply batch normalization? Default yes",
                        type=str, default="yes", required=False,
                        choices=['yes', 'Yes', 'y', 'Y', 'no', 'No', 'n', 'N']
                        )
    args = parser.parse_args()

    DATASET = args.dataset
    BATCH_SIZE = args.batch_size
    WORKERS = args.workers
    GPU = args.gpu
    WEIGHTS = args.weights
    INPUT_TYPE = args.input_type
    BATCH_NORM = args.bn in ['yes', 'Yes', 'y', 'Y']




def eval():

    nyu_classes = ["ceil", "floor", "wall", "window", "chair", "bed", "sofa", "table", "tvs", "furniture", "objects",
                   "empty"]

    print("Selected device:", "cuda:" + str(GPU))
    dev = get_device("cuda:" + str(GPU))
    torch.cuda.empty_cache()

    print('PREPROC_PATH:', PREPROC_PATH)


    if DATASET == "NYUCAD":
        # print(DATASET)
        train_ds = NYU(data_root=PREPROC_PATH, split='train', CAD=True, data_augmentation=False,
                       tsdf_resolution='hr', label_resolution='lr')
        valid_ds = NYU(data_root=PREPROC_PATH, split='test', CAD=True, data_augmentation=False,
                       tsdf_resolution='hr', label_resolution='lr')
    else:
        train_ds = NYU(data_root=PREPROC_PATH, split='train', CAD=False, data_augmentation=False,
                       tsdf_resolution='hr', label_resolution='lr')
        valid_ds = NYU(data_root=PREPROC_PATH, split='test', CAD=False, data_augmentation=False,
                       tsdf_resolution='hr', label_resolution='lr')
    dataloader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    miou = MIoU(num_classes=12, ignore_class=0)
    ciou = CompletionIoU()

    model=SDNET(residual=True, batch_norm=True, inst_norm=False, priors=True)
    print(model)
    if WEIGHTS != "none":
        print("Loading {} ...".format(WEIGHTS))
        # model.load_state_dict(torch.load(WEIGHTS))
        weights_dict = torch.load(WEIGHTS, map_location=torch.device('cpu'))
        # print(weights_dict.keys())
        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)
        print("done")

    model.to(dev)
    model.eval()

    with torch.no_grad():
        with tqdm(total=len(dataloader), desc="") as pbar:
            for sample in dataloader:
                tsdf=sample['tsdf'].to(dev)
                depth=sample['depth_prior'].to(dev)
                gt = sample['label3d']['1'].to(dev)
                label_weight = sample['label_weight']['1'].to(dev)
                files=sample['file']

                pred_ssc,aux_ss= model(tsdf,depth)

                miou.update(pred_ssc, gt, label_weight)
                ciou.update(pred_ssc, gt, label_weight)
                #
                pbar.set_description('Test miou:{:5.2f}'.format(miou.compute() * 100))
                pbar.update()

            comp_iou, precision, recall = ciou.compute()

            print("prec rec. IoU  MIou")
            print("{:4.2f} {:4.2f} {:4.2f} {:4.2f}".format(100 * precision, 100 * recall, 100 * comp_iou,
                                                           miou.compute() * 100))

            per_class_iou = miou.per_class_iou()
            for i in range(len(per_class_iou)):
                text = '{:12.12}: {:5.2f}'.format(nyu_classes[i], 100 * per_class_iou[i])
                print(text, end="        ")
                if i % 4 == 3:
                    print()

            print("\nLatex Line:")
            print("{:4.1f} & {:4.1f} & {:4.1f} &".format(100 * precision, 100 * recall, 100 * comp_iou), end=" ")
            for i in range(len(per_class_iou)):
                text = '{:4.1f} &'.format(100 * per_class_iou[i])
                print(text, end=" ")

            print("{:4.1f} \\\\".format(miou.compute() * 100))
# Main Function
def main():
    parse_arguments()
    eval()


if __name__ == '__main__':
    main()
