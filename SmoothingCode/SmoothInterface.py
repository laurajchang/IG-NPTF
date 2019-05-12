import argparse
import sys
import king_smooth as ks
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--data_file_path',action='store', dest='data_file_path', default='',type=str)
parser.add_argument('--save_dir',action='store', dest='save_dir', default='',type=str)
parser.add_argument('--save_tag',action='store', dest='save_tag', default='',type=str)
parser.add_argument("--ebin", action="store", dest="ebin", default=0,type=int)

results = parser.parse_args()
data_file_path = results.data_file_path
save_dir = results.save_dir
save_tag = results.save_tag
ebin = results.ebin

the_map = np.load(data_file_path)["flux_map"]

maps_dir = '/tigress/smsharma/public/CTBCORE/'

ksi = ks.king_smooth(maps_dir,ebin=ebin,eventclass=5,eventtype=3,threads=1)
the_map_smoothed = ksi.smooth_the_map(the_map)

np.save(save_dir+save_tag+".npy", the_map_smoothed)