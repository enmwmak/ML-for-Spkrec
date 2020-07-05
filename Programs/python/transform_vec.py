#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 19:56:36 2019

@author: mwmak

Read the ark files indexed by an scp file. Then, apply IDVC/DANN transformation on the vectors and
save the transformed vectors to an scp and an ark file

Example usage:
  python3 transform_vec.py -x idvc -m data/pkl/idvc.pkl -i ../v2-1/exp_original/xvectors_sre16_dev_spkgrp/xvector.scp \
                  -o tmp/xvectors_sre16_dev_spkgrp -b ../v2-1

  python3 transform_vec.py -x dann -m data/pkl/dann.pkl -i ../v2-1/exp_original/xvectors_sre16_dev_spkgrp/xvector.scp \
                  -o tmp/xvectors_sre16_dev_spkgrp -b ../v2-1 

"""
from idvc import IDVC
from dann import DANN
from util_func import load_scpfile
import argparse
import pathlib
import kaldi_io
import os
from sklearn.manifold import TSNE
from myPlot import scatter2D
import matplotlib.pyplot as plt
from util_func import load_xvectors, select_speakers
import numpy as np


def save_as_ark_scp(data, output_dir, vec_type='xvector'):
    pathlib.Path('/'.join(output_dir.split('/')[:-1])).mkdir(parents=True, exist_ok=True)
    ark_scp_output = 'ark:| /usr/local/kaldi/src/bin/copy-vector ark:- ark,scp:{0}.ark,{0}.scp'.format(output_dir + '/' + vec_type)
    with kaldi_io.open_or_fd(ark_scp_output, 'wb') as f:
        for key, mat in data.items():
            kaldi_io.write_vec_flt(f, mat, key=key)


def main():
    isDebug = False
    parser = argparse.ArgumentParser()
    parser.add_argument("-x", "--transform", help="Type of transformation (idvc/dann)", default="idvc")
    parser.add_argument("-m", "--model", help="File storing transformation object", default="data/pkl/idvc.pkl")
    parser.add_argument("-i", "--in_scp", help="Input Kaldi scp file", default="../v2-1/exp/xvectors_sre18_eval/xvector.scp")
    parser.add_argument("-o", "--out_dir", help="Output dir storing scp and ark files", default="tmp/xvectors_sre18_eval")
    parser.add_argument("-b", "--base_dir", help="Base dir for the dir index in the .scp file", default="../v2-1")
    parser.add_argument("-t", "--vec_type", help="Vector type [ivector|xvector]", default="xvector")

    args = parser.parse_args()
    
    # Load i-vector or x-vector
    key, val = load_scpfile(args.base_dir, args.in_scp, arktype='vec')
    
    # Perform transformation of i/x-vectors
    if args.transform == 'idvc':
        idvc = IDVC.load_model(args.model)
        X = idvc.transform(val)
    elif args.transform == 'dann':
        encoder_wht = DANN.load_model(args.model)
        X = encoder_wht.transform(val)
    else:
        print('Only idvc or dann transformation is supported')
        exit()
    
    # Save transformed vectors to scp and ark files
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    print("Saving ark and scp to %s" % args.out_dir)    
    save_as_ark_scp(dict(zip(key, X)), args.out_dir, vec_type=args.vec_type)
    
    if isDebug:        
        # Debugging: Load transformed scp file and plot t-sne
        X, spk_lbs, dom_lbs, utt_ids = load_xvectors(['sre16_dev_spkgrp'], 'tmp/', 
                                                     args.vec_type, '../v2-1/data/', 
                                                     './', min_n_vecs=10)

        X, _, _, _ = select_speakers(X, spk_lbs, dom_lbs, utt_ids, 
                                                 min_n_vecs=10, n_spks=10)
        V, spk_lbs, dom_lbs, _ = select_speakers(val, spk_lbs, dom_lbs, utt_ids, 
                                                 min_n_vecs=10, n_spks=10)

        # Plot original x-vectors on 2-D t-SNE space
        print('Creating t-SNE plot of x-vectors')
        X_prj = TSNE(random_state=20150101).fit_transform(X)
        fig, _, _, _ = scatter2D(X_prj, spk_lbs, markers=dom_lbs, n_colors=np.max(spk_lbs)+1,
                                 title='Transformed x-vectors')
        plt.show(block=False)
        
        V_prj = TSNE(random_state=20150101).fit_transform(V)
        fig, _, _, _ = scatter2D(V_prj, spk_lbs, markers=dom_lbs, n_colors=np.max(spk_lbs)+1,
                                 title='Original x-vectors')
        plt.show(block=True)


if __name__ == '__main__':
    main()

