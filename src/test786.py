from argparse import ArgumentParser

import torch.utils.data

from data import read_all_data, all_devices
from federated_util import *
from grid_search import run_grid_search
from supervised_data import get_client_supervised_initial_splitting
from test_hparams import test_hyperparameters
from unsupervised_data import get_client_unsupervised_initial_splitting
