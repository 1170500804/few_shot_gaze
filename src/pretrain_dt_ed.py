import h5py
import numpy as np
from collections import OrderedDict
import gc
import json
import time
import os

import moviepy.editor as mpy

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from data import HDFDataset
from models import DTED
from apex import amp
import logging
import torchvision
import argparse
from checkpoints_manager import CheckpointsManager
from tensorboardX import SummaryWriter

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


parser = argparse.ArgumentParser(description='Train DT-ED')
parser.add_argument('--debug' ,action='store_true')

# architecture specification
parser.add_argument('--densenet-growthrate', type=int, default=32,
                    help='growth rate of encoder/decoder base densenet archi. (default: 32)')
parser.add_argument('--z-dim-app', type=int, default=64,
                    help='size of 1D latent code for appearance (default: 64)')
parser.add_argument('--z-dim-gaze', type=int, default=2,
                    help='size of 2nd dim. of 3D latent code for each gaze direction (default: 2)')
parser.add_argument('--z-dim-head', type=int, default=16,
                    help='size of 2nd dim. of 3D latent code for each head rotation (default: 16)')
parser.add_argument('--decoder-input-c', type=int, default=32,
                    help='size of feature map stack as input to decoder (default: 32)')

parser.add_argument('--normalize-3d-codes', action='store_true',
                    help='normalize rows of 3D latent codes')
parser.add_argument('--normalize-3d-codes-axis', default=1, type=int, choices=[1, 2, 3],
                    help='axis over which to normalize 3D latent codes')

parser.add_argument('--triplet-loss-type', choices=['angular', 'euclidean'],
                    help='Apply triplet loss with selected distance metric')
parser.add_argument('--triplet-loss-margin', type=float, default=0.0,
                    help='Triplet loss margin')
parser.add_argument('--triplet-regularize-d-within', action='store_true',
                    help='Regularize triplet loss by mean within-person distance')

parser.add_argument('--all-equal-embeddings', action='store_true',
                    help='Apply loss to make all frontalized embeddings similar')

parser.add_argument('--embedding-consistency-loss-type',
                    choices=['angular', 'euclidean'], default=None,
                    help='Apply embedding_consistency loss with selected distance metric')
parser.add_argument('--embedding-consistency-loss-warmup-samples',
                    type=int, default=1000000,
                    help='Start from 0.0 and warm up embedding consistency loss until n samples')

parser.add_argument('--backprop-gaze-to-encoder', action='store_true',
                    help='Add gaze loss term to single loss and backprop to entire network.')

parser.add_argument('--coeff-l1-recon-loss', type=float, default=1.0,
                    help='Weight/coefficient for L1 reconstruction loss term')
parser.add_argument('--coeff-gaze-loss', type=float, default=0.1,
                    help='Weight/coefficient for gaze direction loss term')
parser.add_argument('--coeff-embedding_consistency-loss', type=float, default=2.0,
                    help='Weight/coefficient for embedding_consistency loss term')

parser.add_argument('--coeff-perceptual-loss', type=float, default=1.0,
                    help='Weight/coefficient for embedding_consistency loss term')

# training
parser.add_argument('--pick-exactly-per-person', type=int, default=None,
                    help='Pick exactly this many entries per person for training.')
parser.add_argument('--pick-at-least-per-person', type=int, default=400,
                    help='Only pick person for training if at least this many entries.')
parser.add_argument('--use-apex', action='store_true',
                    help='Use half-precision floating points via the apex library.')
parser.add_argument('--base-lr', type=float, default=0.00005, metavar='LR',
                    help='learning rate (to be multiplied with batch size) (default: 0.00005)')
parser.add_argument('--warmup-period-for-lr', type=int, default=1000000, metavar='LR',
                    help=('no. of data entries (not batches) to have processed '
                          + 'when stopping  gradual ramp up of LR (default: 1000000)'))
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='training batch size (default: 128)')
parser.add_argument('--decay-interval', type=int, default=0, metavar='N',
                    help='iterations after which to decay the learning rate (default: 0)')
parser.add_argument('--decay', type=float, default=0.8, metavar='decay',
                    help='learning rate decay multiplier (default: 0.8)')
parser.add_argument('--num-training-epochs', type=float, default=20, metavar='N',
                    help='number of steps to train (default: 20)')
parser.add_argument('--l2-reg', type=float, default=1e-4,
                    help='l2 weights regularization coefficient (default: 1e-4)')
parser.add_argument('--print-freq-train', type=int, default=20, metavar='N',
                    help='print training statistics after every N iterations (default: 20)')
parser.add_argument('--print-freq-test', type=int, default=5000, metavar='N',
                    help='print test statistics after every N iterations (default: 5000)')
parser.add_argument('--optimizer', type=str, default='SGD')

# data
# TODO: change file
parser.add_argument('--mpiigaze-file', type=str, default='../preprocess/outputs/MPIIGaze.h5',
                    help='Path to MPIIGaze dataset in HDF format.')
parser.add_argument('--gazecapture-file', type=str, default='../preprocess/outputs/GazeCapture.h5',
                    help='Path to GazeCapture dataset in HDF format.')
parser.add_argument('--test-subsample', type=float, default=1.0,
                    help='proportion of test set to use (default: 1.0)')
parser.add_argument('--num-data-loaders', type=int, default=1, metavar='N',
                    help='number of data loading workers (default: 0)')

parser.add_argument('--dataset', type=str, default="MPIIGaze")
parser.add_argument('--file-path', type=str, default="/data_b/lius/mpii/MPIIGaze.h5")

# logging
parser.add_argument('--use-tensorboard', action='store_true', default=False,
                    help='create tensorboard logs (stored in the args.save_path directory)')

# TODO: change log dir  done
parser.add_argument('--save-path', type=str, default='/data_b/lius/pretrain_dt_ed',
                    help='path to save network parameters (default: .)')
parser.add_argument('--show-warnings', action='store_true', default=False,
                    help='show default Python warnings')

args = parser.parse_args()
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class RunningStatistics(object):
    def __init__(self):
        self.losses = OrderedDict()

    def add(self, key, value):
        if key not in self.losses:
            self.losses[key] = []
        self.losses[key].append(value)

    def means(self):
        return OrderedDict([
            (k, np.mean(v)) for k, v in self.losses.items() if len(v) > 0
        ])

    def reset(self):
        for key in self.losses.keys():
            self.losses[key] = []

if __name__ == '__main__':


    if args.use_tensorboard:
        tensorboard = SummaryWriter(log_dir=args.save_path)

    network = DTED(
        growth_rate=args.densenet_growthrate,
        z_dim_app=args.z_dim_app,
        z_dim_gaze=args.z_dim_gaze,
        z_dim_head=args.z_dim_head,
        decoder_input_c=args.decoder_input_c,
        normalize_3d_codes=args.normalize_3d_codes,
        normalize_3d_codes_axis=args.normalize_3d_codes_axis,
        use_triplet=args.triplet_loss_type is not None,
        backprop_gaze_to_encoder=args.backprop_gaze_to_encoder,
    )
    # vgg16 = torchvision.models.vgg16_bn(pretrained=True)

    print("Initalize DT_ED Model ...")
    saver = CheckpointsManager(network, args.save_path)

    logging.info(network)
    network = network.to(device)

    if args.optimizer == "SGD":
        optimizer = optim.SGD(
            [p for n, p in network.named_parameters() if not n.startswith('gaze')],
            lr=args.base_lr, momentum=0.9,
            nesterov=True, weight_decay=args.l2_reg,
        )
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(
            [p for n, p in network.named_parameters() if not n.startswith('gaze')],
            lr=args.base_lr, momentum=0.9, weight_decay=args.l2_reg,
        )
    else:
        raise NotImplementedError


    if args.use_apex:
        network, optimizer = amp.initialize(network, optimizer,opt_level='O1')


    if torch.cuda.device_count() > 1:
        logging.info('Using %d GPUs!' % torch.cuda.device_count())
        network = nn.DataParallel(network)



    if args.dataset == 'MPIIGaze':
        args.file_path = '/data_b/lius/mpii/MPIIGaze.h5'
        prefixes = ['p00', 'p01', 'p02', 'p03', 'p04', 'p05', 'p06', 'p07', 'p08', 'p09', 'p10', 'p11',
                    'p12', 'p13', 'p14']
        split_index = 10
        train_prefixes = prefixes[:split_index]
        valid_prefixes = prefixes[split_index:]


    elif args.dataset == 'IDIAP':
        args.file_path = "/data_b/lius/idiap/NormalizeIdiap/idiap.h5"
        prefixes = ['13_B_FT_S', '12_B_FT_S', '14_B_FT_S', '15_B_FT_S', '16_B_FT_S']
        split_index = 4
        train_prefixes = prefixes[:split_index]
        valid_prefixes = prefixes[split_index:]


    elif args.dataset == 'UT':
        args.file_path = "/data_b/NormalizeUT/UtMultiview_s00_s14_64.h5"
        prefixes = ['s' + str(i).zfill(2) for i in range(0, 15)]
        split_index = 10
        train_prefixes = prefixes[:split_index]
        valid_prefixes = prefixes[split_index:]

    logging.info("train_prefixes: " + str(train_prefixes))

    criterion_L1 = nn.L1Loss(reduction="mean")
    criterion_L2 = nn.MSELoss(reduction="mean")

    print('Initalize Datasets and Dataloader ...')

    train_dataset = HDFDataset(hdf_file_path=args.file_path,
                               prefixes=train_prefixes,
                               get_2nd_sample=True,
                               pick_exactly_per_person=args.pick_exactly_per_person,
                               pick_at_least_per_person=args.pick_at_least_per_person,
                               )
    valid_dataset = HDFDataset(hdf_file_path=args.file_path,
                               prefixes=valid_prefixes,
                               get_2nd_sample=True,
                               pick_exactly_per_person=args.pick_exactly_per_person,
                               pick_at_least_per_person=args.pick_at_least_per_person,
                               )

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=args.num_data_loaders,
                                  pin_memory=True,
                                  # worker_init_fn=worker_init_fn,
                                  )

    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=args.num_data_loaders,
                                  pin_memory=True,
                                  # worker_init_fn=worker_init_fn,
                                  )

    print('Datasets and Dataloader Done ...')


    def send_data_dict_to_gpu(data):
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.detach().to(device, non_blocking=True)
        return data

    running_train_losses = RunningStatistics()
    running_valid_losses = RunningStatistics()
    running_timings = RunningStatistics()

    print("########################")
    print("Begin Training Step, total training epoch: {}".format(args.num_training_epochs))
    ######################
    # Training step update

    network.train()
    global_step = 0
    total_step = int(args.num_training_epochs * len(train_dataset) / args.batch_size)
    global_valid_step = 0

    for epoch in range(args.num_training_epochs):

        for train_step, samples in enumerate(train_dataloader):
            optimizer.zero_grad()
            if args.debug:
                print("samples type: {}".format(type(samples)))
            # # forward + backward + optimize
            # time_forward_start = time.time()

            samples = send_data_dict_to_gpu(samples)
            is_pretrain = ('image_b' in samples)
            if is_pretrain:
                output_dict = network(samples)

                l1_loss = 0.25 * (criterion_L1(output_dict['r_a'], samples['image_a']) + \
                    criterion_L1(output_dict['r_b'], samples['image_b']) + \
                    criterion_L1(output_dict['s_a'], samples['image_a']) + \
                    criterion_L1(output_dict['s_b'], samples['image_b']))

                loss_to_optimize = args.coeff_l1_recon_loss * l1_loss

                if args.use_apex:
                    with amp.scale_loss(loss_to_optimize, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss_to_optimize.backward()

                optimizer.step()
                running_train_losses.add('L1_loss', l1_loss.item())
                tensorboard.add_scalar('l1_loss_per_train_step', l1_loss.item(), global_step+1)

                print("[{}/{}]step； training loss:{} \tlr: {}".format(global_step, total_step, loss_to_optimize.item(),
                      optimizer.param_groups[0]["lr"]))

            else:
                pass

            global_step += 1


        for k, v in running_train_losses.means().items():
            tensorboard.add_scalar(k + '_per_epoch', v, epoch + 1)

            print("[{}/{}]epoch； {} loss:{} \t".format(epoch, args.num_training_epochs,k, v))

        running_train_losses.reset()

            # time_backward_end = time.time()
            #
            # running_timings.add('forward_and_backward', time_backward_end - time_forward_start)

        if epoch % 10 == 0:
            saver.save_checkpoint(epoch)

        ####################################
        # Test for particular validation set
            print("########################")
            print("Begin Validating Step ...")
            network.eval()
            for valid_step, valid_samples in enumerate(valid_dataloader):
                valid_samples = send_data_dict_to_gpu(valid_samples)
                is_pretrain = ('image_b' in valid_samples)
                if is_pretrain:
                    output_dict = network(valid_samples)

                    l1_loss = 0.25 * (criterion_L1(output_dict['r_a'], valid_samples['image_a']) + \
                                      criterion_L1(output_dict['r_b'], valid_samples['image_b']) + \
                                      criterion_L1(output_dict['s_a'], valid_samples['image_a']) + \
                                      criterion_L1(output_dict['s_b'], valid_samples['image_b']))

                    # TODO； define multiple losses
                    loss_to_optimize = args.coeff_l1_recon_loss * l1_loss

                    if args.use_apex:
                        with amp.scale_loss(loss_to_optimize, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss_to_optimize.backward()

                    optimizer.step()
                    running_valid_losses.add('L1_loss', l1_loss.item())
                    tensorboard.add_scalar('l1_loss_per_valid_step', l1_loss.item(),  global_valid_step + 1)
                else:
                    pass

                global_valid_step += 1

            for k, v in running_valid_losses.means().items():
                tensorboard.add_scalar(k + '_per_epoch', v, epoch/10 + 1)
                print("[{}/{}]epoch； {} valid loss:{} \t".format(epoch, args.num_training_epochs, k, v))


            running_valid_losses.reset()

