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
import torch.nn.functional as F

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



def angular_error(a, b):
    """Calculate angular error (via cosine similarity)."""
    a = pitchyaw_to_vector(a) if a.shape[1] == 2 else a
    b = pitchyaw_to_vector(b) if b.shape[1] == 2 else b

    ab = np.sum(np.multiply(a, b), axis=1)
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)

    # Avoid zero-values (to avoid NaNs)
    a_norm = np.clip(a_norm, a_min=1e-6, a_max=None)
    b_norm = np.clip(b_norm, a_min=1e-6, a_max=None)

    similarity = np.divide(ab, np.multiply(a_norm, b_norm))
    similarity = np.clip(similarity, a_min=-1.0 + 1e-6, a_max=1.0 - 1e-6)

    return np.degrees(np.arccos(similarity))

def pitchyaw_to_vector(pitchyaws):
    n = pitchyaws.shape[0]
    sin = np.sin(pitchyaws)
    cos = np.cos(pitchyaws)
    out = np.empty((n, 3))
    out[:, 0] = np.multiply(cos[:, 0], sin[:, 1])
    out[:, 1] = sin[:, 0]
    out[:, 2] = np.multiply(cos[:, 0], cos[:, 1])
    return out

def nn_angular_error(y, y_hat):
    sim = F.cosine_similarity(y, y_hat, eps=1e-6)
    sim = F.hardtanh(sim, -1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(sim) * (180 / np.pi)


def nn_mean_angular_loss(y, y_hat):
    return torch.mean(nn_angular_error(y, y_hat))

def inference(loader, simclr_model, device, args):
    feature_vector = []
    labels_vector = []
    for step, entry in enumerate(loader):
        x = entry['image_a']
        x = x.to(device)
        y_gaze = entry['gaze']
        y_head = entry['head']
        if args.head:
            y = torch.cat([y_gaze, y_head], dim=1)
        else:
            y = y_gaze
        # get encoding
        with torch.no_grad():
            h, _, z, _ = simclr_model(x, x)
        if args.proj:
            h = z
        h = torch.nn.functional.normalize(h, dim=1)
        h = h.detach()

        feature_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")
    # feature_vector = torch.nn.functional.normalize(feature_vector, dim=1)
    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector


def get_features(simclr_model, train_loader, test_loader, device):
    train_X, train_y = inference(train_loader, simclr_model, device, args)
    test_X, test_y = inference(test_loader, simclr_model, device, args)
    return train_X, train_y, test_X, test_y


def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test, batch_size):
    train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=False
    )

    test = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader

def knn(args, train_X, train_y, test_X, test_y):
    print(train_X.shape)
    print(test_X.shape)
    scores = np.dot(train_X, test_X.T)
    print(scores.shape)
    take_num = args.knn_num
    inferences = None
    error = 0
    for k in range(scores.shape[1]):
        if k % 1000 == 0:
            print('processing {}th instance'.format(k))
        similarity = scores[:,k]
        index = similarity.argsort()[::-1]

        inf = np.mean(train_y[index[:take_num],:3], axis=0)
        inf = inf.reshape(1,3)
        if inferences is None:
            inferences = inf
            print('inferences is None {}'.format(k))
        else:
            inferences = np.r_[inferences, inf]

    error  = angular_error(inferences, test_y[:,:3].reshape(-1,3))
    error = np.mean(error)
    # error/=scores.shape[1]
    return error


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
    # validing step

    network.eval()




