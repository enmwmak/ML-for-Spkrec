"""
Example:
python wrt2ark_scp.py ali.1.txt ali.1
"""
import os
import kaldi_io
import numpy as np
import argparse
import pathlib2 as pathlib

def alignment_to_vad(in_textfile, extend=False):
    # get info
    vad_info = {}
    with open(in_textfile, 'rt') as f:
        alifile = f.readlines()
    alifile = [x.strip() for x in alifile]

    for line in alifile:
        field = line.split(' ')
        segmentid = field[0]
        val = [int(i) for i in field[1:]]

        # Extend the list val[] by inserting first element to its front and
        # appending the last element to its end
        if extend == True:
            val.insert(0, val[0])
            val.append(val[-1])           
        for i in range(len(val)):
            val[i] = val[i] - 1

        # Check if all frames are silence '0'. If yes, consider the whole utt as speech
        if all(v==0 for v in val) == True:
            val[:] = [1]*len(val)
                    
        vad_info[segmentid] = val
    return vad_info

# Save dict data as .ark and .scp files in the ark_scp_dir
def save_as_ark_scp(vad_info, output_dir):
    pathlib.Path('/'.join(output_dir.split('/')[:-1])).mkdir(parents=True, exist_ok=True)
    ark_scp_output = 'ark:| /usr/local/kaldi/src/bin/copy-vector ark:- ark,scp:{0}.ark,{0}.scp'.format(output_dir)
    with kaldi_io.open_or_fd(ark_scp_output, 'wb') as f:
        for key in vad_info.keys():
            val = np.asarray(vad_info[key], dtype='float32')
            kaldi_io.write_vec_flt(f, val, key=key)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_txtfile",
                        help=("Input alignment file, e.g., "
                              "ali.1.txt"))
    parser.add_argument("out_ark_scp_file",
                        help=("Output file in .ark and .scp format, e.g., "
                              "ali.1"))
    args = parser.parse_args()
    vad_info = alignment_to_vad(args.in_txtfile, extend=True)
    save_as_ark_scp(vad_info, args.out_ark_scp_file)


if __name__ == '__main__':
    main()
