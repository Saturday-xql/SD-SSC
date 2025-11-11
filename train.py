import argparse
import os
from torch.utils.data import DataLoader
from utils.data3d import NYU
import torch
from torch import optim
from utils.train_utils import setup_logger,seed_torch
from utils.losses import weightedssccrossentropy
from utils.cuda import get_device
from tqdm import tqdm
import time
from utils.metrics import MIoU, CompletionIoU
import torch.nn as nn
from model.SDNET import SDNET

torch.cuda.empty_cache()
# torch.autograd.set_detect_anomaly(True)
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# default settings
GPU = 0
R=2025
# Optimizer settings
BASE_LR = 1e-3
LR_MULTIPLIER = 5
DECAY = 1e-5
EPOCHS = 200

# Dataloader settings
BATCH_SIZE = 1
VAL_BATCH_MULT = 1
WORKERS = 2


# Model settings
WEIGHTS="none"
BATCH_NORM = True
INPUT_TYPE = "depth+tsdf"
CLASS_BAL = 'Y'
ONE_CYCLE = True

DATA_AUG = 'y'
EXPR_NAME='SDNET'
TSDF_RESOLUTION='hr'
LABEL_RESOLUTION='lr'
PATIENCE=30

DATASET = "NYU"
PREPROC_PATH = '/your_SSC_datasets_path/SSC_datasets/NYU'
LOG_FILE='./weights/R{:04d}_train_{}_{}_{}.log'.format(R,EXPR_NAME,INPUT_TYPE,DATASET)

seed_torch(2025)


def main(args):
    global GPU, BASE_LR, LR_MULTIPLIER, DECAY, EPOCHS, \
        BATCH_SIZE, VAL_BATCH_MULT, WORKERS, DATASET, PREPROC_PATH, \
        WEIGHTS, BATCH_NORM, INPUT_TYPE, CLASS_BAL, ONE_CYCLE, DATA_AUG, EXPR_NAME, \
        TSDF_RESOLUTION, LABEL_RESOLUTION, PATIENCE, LOG_FILE,script_content

    LOG_FILE = args.log_file
    logger = setup_logger(LOG_FILE)


    DATASET = args.dataset
    EXPR_NAME = args.expr_name
    BATCH_SIZE = args.batch_size
    VAL_BATCH_MULT = args.val_batch_multiplier
    BASE_LR = args.base_lr
    LR_MULTIPLIER = args.lr_multiplier
    DECAY = args.decay
    WORKERS = args.workers
    GPU = args.gpu
    WEIGHTS = args.weights
    EPOCHS = args.epochs
    INPUT_TYPE = args.input_type
    TSDF_RESOLUTION = args.tsdf_resolution
    LABEL_RESOLUTION = args.label_resolution
    PATIENCE = args.patience
    BATCH_NORM = args.bn in ['yes', 'Yes', 'y', 'Y']
    CLASS_BAL = args.class_bal in ['yes', 'Yes', 'y', 'Y']
    ONE_CYCLE = args.one_cycle in ['yes', 'Yes', 'y', 'Y']
    DATA_AUG = args.data_aug in ['yes', 'Yes', 'y', 'Y']



    suffix = "{}_{}_{}_{}2{}".format(EXPR_NAME, INPUT_TYPE, DATASET,TSDF_RESOLUTION,LABEL_RESOLUTION)
    if not BATCH_NORM:
        suffix = suffix + "-nobn"
    if not CLASS_BAL:
        suffix = suffix + "-nocb"
    if DATA_AUG:
        logger.info("3D Data augmentation activated!!!")
        suffix = suffix + "_da"

    logger.info("Selected device: cuda:" + str(GPU))
    dev = get_device("cuda:" + str(GPU))
    # torch.cuda.empty_cache()


    if DATASET == "NYUCAD":
        # print(DATASET)
        train_ds = NYU(data_root=PREPROC_PATH, split='train', CAD=True, data_augmentation=DATA_AUG,
                       tsdf_resolution=TSDF_RESOLUTION, label_resolution=LABEL_RESOLUTION)
        valid_ds = NYU(data_root=PREPROC_PATH, split='test', CAD=True, data_augmentation=False,
                       tsdf_resolution=TSDF_RESOLUTION, label_resolution=LABEL_RESOLUTION)
    else:
        train_ds = NYU(data_root=PREPROC_PATH, split='train', CAD=False, data_augmentation=DATA_AUG,
                       tsdf_resolution=TSDF_RESOLUTION, label_resolution=LABEL_RESOLUTION)
        valid_ds = NYU(data_root=PREPROC_PATH, split='test', CAD=False, data_augmentation=False,
                       tsdf_resolution=TSDF_RESOLUTION, label_resolution=LABEL_RESOLUTION)

    logger.info("Train: {}, Valid: {}".format(len(train_ds), len(valid_ds)))


    dataloaders = {
            'train': DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS),
            'valid': DataLoader(valid_ds, batch_size=BATCH_SIZE * VAL_BATCH_MULT, shuffle=False, num_workers=WORKERS)
        }

    logger.info("Input type: " + INPUT_TYPE)

    model = SDNET(residual=True, batch_norm=True, inst_norm=False, priors=True)
    logger.info(model)

    model.to(dev)
    if WEIGHTS != "none":
        logger.info("Loading {} ...".format(WEIGHTS))
        weights_dict = torch.load(WEIGHTS)
        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)
        logger.info("done")



    opt = optim.Adam(model.parameters(), lr=BASE_LR, weight_decay=DECAY, betas=(0.9, 0.999))
    if ONE_CYCLE:
        sch = optim.lr_scheduler.OneCycleLR(opt, max_lr=BASE_LR * LR_MULTIPLIER,
                                        steps_per_epoch=len(dataloaders['train']), epochs=EPOCHS)
    else:
        sch = optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.5, last_epoch=-1)

    class_weights1 = [1, 1] if CLASS_BAL else [1, 1]
    class_weights2 = [0.1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 1, 2] if CLASS_BAL else [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


    criterion_aux_ss = weightedssccrossentropy(class_weight=torch.Tensor(class_weights2).to(dev),ignore_index=0).to(dev)
    criterion_ssc =weightedssccrossentropy(class_weight=torch.Tensor(class_weights2).to(dev),ignore_index=255).to(dev)

    # Train and evaluate
    model = train_3d(model, dev, dataloaders, criterion_aux_ss, criterion_ssc, opt,
                        scheduler=sch, num_epochs=EPOCHS, patience=PATIENCE, suffix=suffix,R=R,logger=logger)



def train_3d(model, dev, dataloaders,criterion_aux_ss, criterion_ssc, optimizer, scheduler=None, num_epochs=25, patience=50,
             suffix=None, R=0, logger=None):
    nyu_classes = ["ceil", "floor", "wall", "window", "chair", "bed", "sofa", "table", "tvs", "furniture", "objects",
                   "empty"]
    run = '{:04d}'.format(R)  # get_run()
    time.sleep(.1)

    if suffix is None:
        model_name = "R{}_{}".format(run,  type(model).__name__)
    else:
        model_name = "R{}_{}".format(run,  suffix)

    logger.info("Training model {}".format(model_name))


    since = time.time()
    best_miou = 0
    waiting = 0
    min_loss=100.0
    for epoch in range(1,num_epochs+1):
        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            aux_sc_loss=0.0
            aux_ss_loss = 0.0
            ssc_loss=0.0

            # Iterate over data.
            tqdm_desc = "{}: Epoch {}/{} Loss: {:.4f} MIoU: {:.4f} Lr: {:.8f} aux_ss_loss: {:.4f} ssc_loss: {:.4f}"
            num_samples = 0
            m_miou = MIoU(num_classes=12, ignore_class=0)
            ciou = CompletionIoU()

            with tqdm(total=len(dataloaders[phase]), desc="") as pbar:
                for batch_num, sample in enumerate(dataloaders[phase]):
                    tsdf = sample['tsdf'].to(dev)
                    label_weight=sample['label_weight']
                    gt = sample['label3d']
                    depth=sample['depth_prior'].to(dev)
                    # print(tsdf.shape)
                    num_samples += tsdf.size(0)

                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        pred_ssc,aux_ss= model(tsdf,depth)

                        gt_ss={}
                        for k, v in gt.items():
                            gt[k]=v.to(dev)
                            label_weight[k] = label_weight[k].to(dev)

                            v_ss=v.clone()
                            v_ss[v_ss==255]=0
                            gt_ss[k]=v_ss.to(dev)

                        loss_aux_ss=0.0
                        for k in gt.keys():
                            loss_aux_ss+=criterion_aux_ss(aux_ss[k], gt_ss[k], label_weight[k])

                        loss_aux_ss/=3
                        loss_ssc = criterion_ssc(pred_ssc, gt['1'], label_weight['1'])

                        loss =  loss_ssc+loss_aux_ss
                        m_miou.update(pred_ssc, gt['1'], label_weight['1'])
                        ciou.update(pred_ssc, gt['1'], label_weight['1'])

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                            if type(scheduler) is optim.lr_scheduler.OneCycleLR:
                                scheduler.step()

                    l = loss.item()
                    running_loss += l * tsdf.size(0)

                    aux_ss_loss += (loss_aux_ss.item()) * tsdf.size(0)
                    ssc_loss += (loss_ssc.item()) * tsdf.size(0)
                    pbar.set_description(tqdm_desc.format(phase, epoch , num_epochs,
                                                          running_loss / num_samples,
                                                          m_miou.compute(),
                                                          optimizer.param_groups[0]['lr'],
                                                          aux_ss_loss / num_samples,
                                                          ssc_loss / num_samples,))
                    pbar.update()

            logger.info(tqdm_desc.format(phase, epoch, num_epochs,
                                                  running_loss / num_samples,
                                                  m_miou.compute(),
                                                  optimizer.param_groups[0]['lr'],
                                                  aux_ss_loss / num_samples,
                                                  ssc_loss / num_samples ))

            epoch_loss = running_loss / num_samples
            epoch_miou = m_miou.compute()
            epoch_per_class_iou = m_miou.per_class_iou()
            comp_iou, precision, recall = ciou.compute()

            if phase == 'train':
                if type(scheduler) is not optim.lr_scheduler.OneCycleLR:
                    scheduler.step()


            if phase == 'valid' and epoch_miou > best_miou:
                waiting = 0
                logger.info("mIoU improved from {:.5f} to {:.5f}".format(best_miou, epoch_miou))
                best_epoch = epoch
                best_miou = epoch_miou
                best_per_class_iou = epoch_per_class_iou
                torch.save(model.state_dict(), "weights/{}".format(model_name))
                if min_loss > epoch_loss:
                    min_loss=epoch_loss
            elif phase == 'valid' and min_loss > epoch_loss:
                waiting=0
                min_loss=epoch_loss
                logger.info("mIoU {:.5f} was not an improvement from {:.5f}".format(epoch_miou, best_miou))
                torch.save(model.state_dict(), "weights/{}_EPOCH_{}".format(model_name, "new"))
            elif phase == 'valid':
                logger.info("mIoU {:.5f} was not an improvement from {:.5f}".format(epoch_miou, best_miou))
                waiting += 1
                torch.save(model.state_dict(), "weights/{}_EPOCH_{}".format(model_name, "new"))

            if phase == 'valid':
                tb_text = "\nEpoch:{:3d}\n".format(epoch)
                tb_text += "prec rec. IoU  MIou\n"
                tb_text += "{:4.2f} {:4.2f} {:4.2f} {:4.2f}\n".format(100 * precision, 100 * recall, 100 * comp_iou,
                                                                      epoch_miou * 100)

                for i in range(11):
                    text = '{:12.12}: {:5.2f}        '.format(nyu_classes[i], 100 * epoch_per_class_iou[i])
                    tb_text += text
                    if ((i + 1) % 3 == 0):
                        tb_text += '\n'
                logger.info(tb_text + "\n\n")
                time.sleep(.5)

        if waiting > patience:
            logger.info("Out of patience!!!")
            break

    torch.save(model.state_dict(), "weights/{}_EPOCH_{}".format(model_name, epoch))
    time_elapsed = time.time() - since
    logger.info(model_name)
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logger.info('Best val MIoU%: {:6.2f}  Epoch: {}'.format(100 * best_miou, best_epoch))

    tb_text = "\n"
    for i in range(11):
        text = '{:12.12}: {:5.2f}        '.format(nyu_classes[i], 100 * best_per_class_iou[i])
        tb_text += text
        if ((i + 1) % 3 == 0):
            tb_text += '\n'
    logger.info(tb_text + "\n\n")
    return model



if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Target dataset", type=str, default=DATASET, choices=['NYU', 'NYUCAD'])
    parser.add_argument("--batch_size", help="Training batch size. Default: " + str(BATCH_SIZE),
                        type=int, default=BATCH_SIZE, required=False)
    parser.add_argument("--val_batch_multiplier", help="Val batch size. Default: " + str(VAL_BATCH_MULT),
                        type=int, default=VAL_BATCH_MULT, required=False)
    parser.add_argument("--base_lr", help="Base LR for One cycle learning. Default " + str(BASE_LR),
                        type=float, default=BASE_LR, required=False)
    parser.add_argument("--lr_multiplier", help="Max LR multiplier. Default " + str(LR_MULTIPLIER),
                        type=float, default=LR_MULTIPLIER, required=False)
    parser.add_argument("--decay", help="Weight decay. Default " + str(DECAY),
                        type=float, default=DECAY, required=False)
    parser.add_argument("--workers", help="Concurrent threads. Default " + str(WORKERS),
                        type=int, default=WORKERS, required=False)
    parser.add_argument("--gpu", help="GPU device. Default " + str(GPU),
                        type=int, default=GPU, required=False)
    parser.add_argument("--weights", help="Pretraind weights. Default " + WEIGHTS,
                        type=str, default=WEIGHTS, required=False)
    parser.add_argument("--epochs", help="How many epochs? Default " + str(EPOCHS),
                        type=int, default=EPOCHS, required=False)
    parser.add_argument("--input_type", help="Network input type. Default " + INPUT_TYPE,
                        type=str, default=INPUT_TYPE, required=False,
                        choices=['rgb+normals', 'rgb+depth', 'depth', 'tsdf']
                        )
    parser.add_argument("--tsdf_resolution", help="tsdf resolution", type=str, default=TSDF_RESOLUTION,
                        choices=['lr', 'hr'])
    parser.add_argument("--label_resolution", help="label resolution", type=str, default=LABEL_RESOLUTION,
                        choices=['lr', 'hr'])
    parser.add_argument("--patience", help=" ", type=int, default=PATIENCE, required=False)
    parser.add_argument("--expr_name", help="experiment name", default=EXPR_NAME, type=str)
    parser.add_argument('--log_file', type=str, default=LOG_FILE)

    parser.add_argument("--bn", help="Apply batch bormalization?. Default yes",
                        type=str, default="yes", required=False,
                        choices=['yes', 'Yes', 'y', 'Y', 'no', 'No', 'n', 'N']
                        )
    parser.add_argument("--class_bal", help="Apply class balancing?. Default yes",
                        type=str, default=CLASS_BAL, required=False,
                        choices=['yes', 'Yes', 'y', 'Y', 'no', 'No', 'n', 'N']
                        )
    parser.add_argument("--one_cycle", help="Apply OCL?. Default yes",
                        type=str, default="yes", required=False,
                        choices=['yes', 'Yes', 'y', 'Y', 'no', 'No', 'n', 'N']
                        )
    parser.add_argument("--data_aug", help="Data augmentation?Is it an oracle test?. Default no",
                        type=str, default=DATA_AUG, required=False,
                        choices=['yes', 'Yes', 'y', 'Y', 'no', 'No', 'n', 'N']
                        )
    args = parser.parse_args()
    main(args)



