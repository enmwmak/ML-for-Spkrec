# Convert Matlab .mat to Kaldi's .ark file
# Write key-value pairs to .ark file
# Note that the .mat file should contain two variables of name and types like this
#  Name            Size                Bytes  Class     Attributes
#  X            2332x600            11193600  double              
#  spk_ids      2332x18                83952  char             
# for 'spk_ids', the type should not be cellarray

import kaldi_io
import numpy as np
import argparse
import scipy.io as sio
import pathlib


# Save dict data as .ark and .scp files in the ark_scp_dir
def save_as_ark_scp(data, output_dir):
    pathlib.Path('/'.join(output_dir.split('/')[:-1])).mkdir(parents=True, exist_ok=True)
    ark_scp_output = 'ark:| /usr/local/kaldi/src/bin/copy-vector ark:- ark,scp:{0}.ark,{0}.scp'.format(output_dir)
    with kaldi_io.open_or_fd(ark_scp_output, 'wb') as f:
        for key, mat in data.items():
            kaldi_io.write_vec_flt(f, mat, key=key)


# Convert Matlab .mat file to Kaldi .ark file. The .mat file should
# have two variables: spk_ids and X, which will form the key-value pair in .ark
def mat2ark(matfile, arkfile, arktype='vec'):
    print('Reading %s' % matfile)
    data = sio.loadmat(matfile)
    key = data['spk_ids']
    if np.isscalar(data['X'][0][0]):
        val = list(data['X'])                 # mfcc or ivectors
    else:
        val = list(data['X'][0])              # vad
    print('Writing %s' % arkfile)
    if arktype == 'vec' or arktype == 'ary':  # One vector per key, e.g. i-vector
        with open(arkfile, 'wb') as f:
            for k, v in zip(key, val):
                v = np.reshape(v, (v.shape[-1],))   # Reshape from (1,xxx) to (xxx,)
                kaldi_io.write_vec_flt(f, v, key=k)
    elif arktype == 'mat':              # One matrix per key, e.g. mfcc
        with open(arkfile, 'wb') as f:
            for k, v in zip(key, val):
                kaldi_io.write_mat(f, v, key=k)
    else:
        raise ValueError('Ark type %s not supported' % arktype)


# Convert .mat files to Kaldi .ark files. The mat file should
# have variables 'spk_ids' and 'X'.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", help="Input filename with extension [mat]")
    parser.add_argument("outfile", help="Output filename with extension [ark]")
    parser.add_argument("-i", "--input_type",
                        help="Input file type [mat]",
                        default="mat")
    parser.add_argument("-o", "--output_type", help="Output file type [ark]",
                        default="ark")
    parser.add_argument("-a", "--ark_type", help="Type of ark file [vec|mat|ary]",
                        default="vec")
    args = parser.parse_args()
    if args.input_type == 'mat':
        if args.output_type == 'ark':
            mat2ark(args.infile, args.outfile, arktype=args.ark_type)
        else:
            print('Unsupported output file type %s' % args.output_type)
    else:
        print('Unsupported input file type %s' % args.input_type)


if __name__ == '__main__':
    main()