import enum
import os
import json
import numpy as np
import argparse
from glob import glob
import matplotlib.pyplot as plt
from pandas import DataFrame
import code.curr_polygon_back_sub as curr_polygon_back_sub


parser =  argparse.ArgumentParser(
     description='Start the evaluation process of a particular Method.')
parser.add_argument('--config',type=str,help="path to config dir")
parser.add_argument('--dataset_path', type = str, help="path to dataset")
parser.add_argument('--test_file', type = str, help="path to test file")
parser.add_argument('--output_dir', type=str,
                    help='path to output file.', default="output")
args = parser.parse_args()

print(args.dataset_path)

# assert os.path.exists(args.config), "config_dir does not exist"
# assert os.path.exists(args.dataset_path), "dataset path does not exist"

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)


config = json.load(open(args.config))

if(config["model_type"] == "canny"):

    temp_list = []

    header = ["file_name","Object Type","Number of edge pixels","height","width","ratio","Blurred?"]

    import polygon_canny
    filepath = args.dataset_path
    with open(filepath) as f:
        lines = f.readlines()
        curr_polygon_back_sub.canny_result(lines,config,args)
