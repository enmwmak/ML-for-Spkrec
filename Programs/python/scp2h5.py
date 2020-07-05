#!/usr/local/anaconda3/envs/tfenv/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 10:25:10 2019

@author: mwmak

Convert the .ark files indexed by a .scp file to a .h5 file. 
Because vectors in Kaldi's ark files are indexed by scp file, we need to make sure that
exp/ is linked to exp_original/ before running this program. 

Example for one domain per .h5 file with at least 10 utts per speaker:
    python3 scp2h5.py -d swbd2 -b ../v2-1 -o data/h5/xvectors_swbd2.h5 -e ../v2-1/exp/ -a ../v2-1/data/ -t xvector -m 10
    python3 scp2h5.py -d swbdcell -b ../v2-1 -o data/h5/xvectors_swbdcell.h5 -e ../v2-1/exp/ -a ../v2-1/data/ -t xvector -m 10
    python3 scp2h5.py -d sre04-10-mx6 -b ../v2-1 -o data/h5/xvectors_sre04-10-mx6.h5 -e ../v2-1/exp/ -a ../v2-1/data/ -t xvector -m 10
    python3 scp2h5.py -d sre16_dev_spkgrp -b ../v2-1 -o data/h5/xvectors_sre16_dev_spkgrp.h5 -e ../v2-1/exp/ -a ../v2-1/data/ -t xvector -m 10
    python3 scp2h5.py -d sre16_eval_spkgrp -b ../v2-1 -o data/h5/xvectors_sre16_eval_spkgrp.h5 -e ../v2-1/exp/ -a ../v2-1/data/ -t xvector -m 10
    python3 scp2h5.py -d sitw_eval -b ../v2-1 -o data/h5/xvectors_sitw_eval.h5 -e ../v2-1/exp/ -a ../v2-1/data/ -t xvector -m 10
    python3 scp2h5.py -d voxceleb1 -b ../v2-1 -o data/h5/xvectors_voxceleb1.h5 -e ../v2-1/exp/ -a ../v2-1/data/ -t xvector -m 10
    python3 scp2h5.py -d sre18_dev -b ../v2-1 -o data/h5/xvectors_sre18_dev.h5 -e ../v2-1/exp/ -a ../v2-1/data/ -t xvector -m 10
    python3 scp2h5.py -d sitw_dev -b ../v2-1 -o data/h5/xvectors_sitw_dev.h5 -e ../v2-1/exp/ -a ../v2-1/data/ -t xvector -m 10

Example for multiple-domain per .h5 file:
    python3 scp2h5.py -d sre18_dev_spkgrp,sitw_dev -b ../v2-1 -o data/h5/xvectors_idvc_test.h5 \
        -e ../v2-1/exp/ -a ../v2-1/data/ -t xvector -m 20 
    python3 scp2h5.py -d swbd2,swbdcell,sre04-10-mx6,sre16_spkgrp,sre18_dev_spkgrp \
        -b ../v2-1 -o data/h5/xvectors_idvc_train.h5 -e ../v2-1/exp/ -a ../v2-1/data/ -t xvector  -m 10 
    python3 scp2h5.py -d sre18_dev_spkgrp,sitw_dev \
        -b ../v2-1 -o data/h5/xvectors_idvc_test.h5 -e ../v2-1/exp/ -a ../v2-1/data/ -t xvector  -m 10 

Example for one domain per .h5 file with at least 1 utt per speaker:
    python3 scp2h5.py -d sre18_eval_enroll -b ../v2-1 -o data/h5/xvectors_sre18_eval_enroll -e ../v2-1/exp/ -a ../v2-1/data/ -t xvector -m 1 
    python3 scp2h5.py -d sre18_eval_test -b ../v2-1 -o data/h5/xvectors_sre18_eval_test -e ../v2-1/exp/ -a ../v2-1/data/ -t xvector -m 1 
    
"""
from util_func import load_xvectors, save_xvectors_h5
import argparse
import os
import numpy as np

def main():
    print("Make sure that exp/ is linked to exp_original/")
    os.remove('../v2-1/exp')
    os.symlink('../v2-1/exp_original', '../v2-1/exp')
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--domains", help="Domain or dataset name", default="sre16_dev")
    parser.add_argument("-o", "--out_file", help="Output file", default="data/h5/xvectors_sre16_dev.h5")
    parser.add_argument("-b", "--base_dir", help="Base dir for the dir index in the .scp file", default="../v2-1")
    parser.add_argument("-e", "--exp_dir", help="exp/ dir of Kaldi recipe", default="../v2-1/exp/")
    parser.add_argument("-a", "--data_dir", help="data/ dir of Kaldi recipe", default="../v2-1/data/")
    parser.add_argument("-t", "--vec_type", help="Vector type [ivector|xvector]", default="xvector")
    parser.add_argument("-m", "--min_n_vecs", help="Minimum no. of utts per speaker", default="1")

    args = parser.parse_args()
    domains = args.domains.split(",")
    min_n_vecs = int(args.min_n_vecs)
    X, spk_lbs, dom_lbs, utt_ids = load_xvectors(domains, args.exp_dir, args.vec_type, 
                                                 args.data_dir, args.base_dir, min_n_vecs=min_n_vecs)
    for dom in range(np.max(dom_lbs)+1):
        N = dom_lbs[dom==dom_lbs].shape[0]
        print('No. of vectors from Domain %d = %d' % (dom, N))
    save_xvectors_h5(args.out_file, X, spk_lbs, dom_lbs, utt_ids, domains)

if __name__ == '__main__':
    main()


