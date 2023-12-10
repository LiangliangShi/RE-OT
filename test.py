"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the OLTR project which
notice below and in LICENSE in the root directory of
this source tree.

Copyright (c) 2019, Zhongqi Miao
All rights reserved.
"""

import os
import argparse
import pprint
from data import dataloader
from run_networks_test import model
import warnings
import yaml
from utils import get_value, set_seed


data_root = {
    "ImageNet": "./dataset/imagenet",
    "Places": "./dataset",
    "iNaturalist18": "/checkpoint/bykang/iNaturalist18",
    "CIFAR10": "./dataset",
    "CIFAR100": "./dataset",
}

parser = argparse.ArgumentParser()
parser.add_argument("--cfg", default=None, type=str)
parser.add_argument("--test", default=False, action="store_true")
parser.add_argument("--batch_size", type=int, default=None)
parser.add_argument("--test_open", default=False, action="store_true")
parser.add_argument("--output_logits", default=False)
parser.add_argument("--model_dir", type=str, default=None)
parser.add_argument("--save_feat", type=str, default="")

# KNN testing parameters
parser.add_argument("--knn", default=False, action="store_true")
parser.add_argument("--feat_type", type=str, default="cl2n")
parser.add_argument("--dist_type", type=str, default="l2")

# Learnable tau
parser.add_argument("--val_as_train", default=False, action="store_true")

parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()

set_seed(args.seed)


def update(config, args):
    # Change parameters
    config["model_dir"] = get_value(config["model_dir"], args.model_dir)
    config["training_opt"]["batch_size"] = get_value(config["training_opt"]["batch_size"], args.batch_size)
    config["training_opt"]["seed"] = args.seed

    # Testing with KNN
    if args.knn and args.test:
        training_opt = config["training_opt"]
        classifier_param = {
            "feat_dim": training_opt["feature_dim"],
            "num_classes": training_opt["num_classes"],
            "feat_type": args.feat_type,
            "dist_type": args.dist_type,
            "log_dir": training_opt["log_dir"],
        }
        classifier = {
            "def_file": "./models/KNNClassifier.py",
            "params": classifier_param,
            "optim_params": config["networks"]["classifier"]["optim_params"],
        }
        config["networks"]["classifier"] = classifier

    return config


# ============================================================================
# LOAD CONFIGURATIONS
with open(args.cfg) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
config = update(config, args)

test_mode = args.test
test_open = args.test_open
if test_open:
    test_mode = True
output_logits = args.output_logits
training_opt = config["training_opt"]
relatin_opt = config["memory"]
dataset = training_opt["dataset"]

if not os.path.isdir(training_opt["log_dir"]):
    os.makedirs(training_opt["log_dir"])

print("Loading dataset from: %s" % data_root[dataset.rstrip("_LT")])
pprint.pprint(config)


def split2phase(split):
    if split == "train" and args.val_as_train:
        return "train_val"
    else:
        return split


warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

print(
    "Under testing phase, we load training data simply to calculate \
        training data number for each class."
)

if "iNaturalist" in training_opt["dataset"]:
    splits = ["train", "val"]
    test_split = "val"
else:
    splits = ["train", "val", "test"]
    test_split = "test"
if "ImageNet" == training_opt["dataset"]:
    splits = ["train", "val"]
    test_split = "val"
if args.knn or True:
    splits.append("train_plain")

data = {
    x: dataloader.load_data(
        data_root=data_root[dataset.rstrip("_LT")],
        dataset=dataset,
        phase=x,
        batch_size=training_opt["batch_size"],
        sampler_dic=None,
        test_open=test_open,
        num_workers=training_opt["num_workers"],
        shuffle=False,
        cifar_imb_ratio=training_opt["cifar_imb_ratio"] if "cifar_imb_ratio" in training_opt else None,
    )
    for x in splits
}

training_model = model(config, data, test=True)
training_model.load_model(args.model_dir)
if args.save_feat in ["train_plain", "val", "test"]:
    saveit = True
    test_split = args.save_feat
else:
    saveit = False

training_model.eval(phase=test_split, openset=test_open, save_feat=saveit)

if output_logits:
    training_model.output_logits(openset=test_open)

print("ALL COMPLETED.")
