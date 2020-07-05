"""
Read Kaldi .ark and .scp files and convert to .h5, .csv, and .mat
Require to setup PYTHONPATH as follows:
export PYTHONPATH=$HOME/so/python/kaldi_io
kaldio_io can be downloaded from https://github.com/vesis84/kaldi-io-for-python

The -i option specifies the input data type, which can be 
   ark: Kaldi ark file
   scp: Kaldi scp file
   
The -o option specifies the output data type, which can be
   h5 : Python h5py file
   csv: Delimited text file
   mat: Matlab .mat file   

The -a option specifies the type of data stored in the .ark file. It can be
   mat: The ark file contains a number of matrices, each with the same no. of columns (e.g. MFCC)
   lom: The ark file contains a number of matrices, each with different size (e.g.pairwise PLDA scores)
   vec: The ark file contains a number of vectors, each with the same dim (e.g. i-vectors)
   ary: The ark file contains a number of arrays, each with different length (e.g. VAD)


To use this program to convert VAD file (in .ark) to TIMIT .phn files, execute

  python3 kaldi_arkread.py -i ark -o phn -a ary ../v2-1/mfcc/vad_sre16_dev_enroll.10.ark ../v2-1/phn

This will create a folder ../v2-1/phn, where the .phn files corresponding to the speech files in
the .ark file will be stored.


To use this program to convert the .ark files of a directory, use the .scp that indexes to all of the
.ark files in that directory

  python3 kaldi_arkread.py -i scp -o h5 -a vec exp/xvectors_sre16_dev/xvector.scp data/h5/xvectors_sre16_dev.h5 ../v2-1    

Execute "h5ls data/h5/xvectors_sre16_dev.h5" to read the fields in the data file.
Note that in the above example, the 'spk_ids' are utterance IDs because each entry in 'spk_ids" field 
is suffixed with the utterance name.

"""

import kaldi_io
import numpy as np
import glob, os, re, sys
import h5py as h5
import argparse
import scipy.io as sio


# Stripe or append text in key and truncate columns in matrix
def adjust_key_mat(key, mat, strip_after_chars=None,
                   append_str=None, n_dims_keeped=None):
    if strip_after_chars is not None:
        for i in range(len(key)):
            for ch in strip_after_chars:
                key[i] = key[i].split(ch)[0]
    if append_str is not None:
        for i in range(len(key)):
            key[i] = key[i] + append_str
    if n_dims_keeped is not None:
        mat = mat[:, :n_dims_keeped]
    return key, mat


# Save key and val to h5 file
def save_h5(key, val, h5_file):
    unicode = h5.special_dtype(vlen=str)
    print('Saving %s' % h5_file)
    with h5.File(h5_file, 'w') as f:
        f['spk_ids'] = key.astype(unicode)
        f['X'] = val


# Save key and val to csv file
def save_csv(val, csvfile):
    print('Saving %s' % csvfile)
    if isinstance(val, np.ndarray):
        np.savetxt(csvfile, val, fmt='%.4f', delimiter=' ')
    elif isinstance(val, list):
        with open(csvfile, 'wb') as f:
            for line in val:
                np.savetxt(f, line, fmt='%.4f', delimiter=' ', newline=' ')
                f.write(b"\n")      # Append a newline char
    else:
        print('Can only save 2-D array or list of 1-D array to csv file')
        raise NotImplementedError


# Save key and value to Matlab .mat file
def save_mat(key, val, matfile):
    print('Saving %s' % matfile)
    sio.savemat(matfile, {'spk_ids':key, 'X':val})

# Save key and value to Matlab .mat file. Note that the syntax is the same
# as save_mat but the matrices will be saved as cell of matrices in Matlab    
def save_lom(key, val, matfile):
    print('Saving %s' % matfile)
    sio.savemat(matfile, {'spk_ids':key, 'X':val})

def load_arkfile(arkfile, arktype='vec'):
    print('Reading %s' % arkfile)
    if arktype == 'vec':
        key, val = load_vec_arkfile(arkfile)
    elif arktype == 'mat':
        key, val = load_mat_arkfile(arkfile)
    elif arktype == 'ary':
        key, val = load_ary_arkfile(arkfile)
    elif arktype == 'lom':
        key, val = load_lom_arkfile(arkfile)
    else:
        raise NotImplementedError
    return key, val


def load_scpfile(basedir, scpfile, arktype='vec'):
    print('Reading %s' % scpfile)
    if arktype == 'vec':
        key, val = load_vec_ark_from_scp(basedir, scpfile)
    elif arktype == 'mat':
        key, val = load_mat_ark_from_scp(basedir, scpfile)
    elif arktype == 'ary':
        key, val = load_ary_ark_from_scp(basedir, scpfile)
    else:
        raise NotImplementedError
    return key, val


# Load matrix-type .ark files based on the .scp file
def load_mat_ark_from_scp(basedir, scpfile):
    cwd = os.getcwd()
    os.chdir(basedir)
    key = list()
    val = list()
    for k, v in kaldi_io.read_mat_scp(scpfile):
        key.append(k)
        val.append(v)
    mat = np.vstack(val)
    os.chdir(cwd)
    return key, mat


# Load vector-type .ark files based on the .scp file
# NB: All vectors in .ark files have to be of the same size
def load_vec_ark_from_scp(basedir, scpfile):
    cwd = os.getcwd()
    os.chdir(basedir)
    key = list()
    val = list()
    for k, v in kaldi_io.read_vec_flt_scp(scpfile):
        key.append(k)
        val.append(v)

    # Stack matrices to form one matrix
    mat = np.vstack(val)
    os.chdir(cwd)
    return key, mat


# Load array-type .ark files based on the .scp file
# NB: Array in .ark files could have different size. So, return
# list of array instead of a matrix
def load_ary_ark_from_scp(basedir, scpfile):
    cwd = os.getcwd()
    os.chdir(basedir)
    key = list()
    val = list()
    for k, v in kaldi_io.read_vec_flt_scp(scpfile):
        key.append(k)
        val.append(v)
    os.chdir(cwd)
    return key, val


# Read a single matrix-type .ark file (e.g. mfcc) and return the keys and matrix pairs
# Because all matrices have the same number of columns (e.g., dim of mfcc), we may
# stack the list of matrix into a 2-D numpy array.    
def load_mat_arkfile(arkfile):
    key = []
    mat = []
    for k,v in kaldi_io.read_mat_ark(arkfile):
        key.append(k)
        mat.append(v)
    mat = np.vstack(mat)
    return key, mat

# Read a list of matrix-type .ark file (e.g. pairwise PLDA scores) and return the
# keys and list of matrix pairs. Note that matrices could have different sizes.
def load_lom_arkfile(arkfile):
    key = []
    mat = []
    for k,v in kaldi_io.read_mat_ark(arkfile):
        key.append(k)
        mat.append(v)
    return key, mat
    

# Read a single vector-type .ark file (e.g ivector) and return the key and matrix pair
def load_vec_arkfile(arkfile):
    key = []
    val = []
    for k, v in kaldi_io.read_vec_flt_ark(arkfile):
        key.append(k)
        val.append(v)
    val = np.vstack(val)
    return key, val


# Read a single array-type .ark file (e.g vad) and return the key and array pair
# NB: Array size could be different for each key. So, return list of ndarray
def load_ary_arkfile(arkfile):
    key = []
    val = []
    for k,v in kaldi_io.read_vec_flt_ark(arkfile):
        key.append(k)
        val.append(v)
    return key, val


# Read all .ark files in a given directory. Return the packed key and matrix pairs
# Remark: This way of reading all i-vectors is much faster than
# using the exp/<...>/ivector.scp
def load_arkdir(arkdir, pattern='ivector.*.ark', nfiles=1, arktype='vec'):
    key = list()
    val = list()
    n = 0
    cwd = os.getcwd()
    os.chdir(arkdir)
    for filename in glob.glob('*.ark'):
        if re.match(pattern, filename):
            print('Reading %s/%s' % (arkdir, filename))
            if arktype == 'vec':
                k, v = load_vec_arkfile(filename)
            elif arktype == 'mat':
                k, v = load_mat_arkfile(filename)
            elif arktype == 'lom':
                k, v = load_lom_arkfile(filename)
            key.append(k)
            val.append(v)
            n = n + 1
            if n == nfiles:
                break

    mat = np.vstack(val)

    # Flatten list-of-list to form one list
    flat_key = list()
    for subkey in key:
        for item in subkey:
            flat_key.append(item)
    os.chdir(cwd)
    return flat_key, mat


# Convert all .ark files (indexed by .scp) in a given dir to a single .h5 file
# The .h5 file should contain 'X' and 'spk_ids' fields
def scp2h5(basedir, scpfile, h5file, n_dims_keeped=None,
           strip_after_chars=None, append_str=None, arktype='vec'):
    key, mat = load_scpfile(basedir, scpfile, arktype=arktype)
    adjust_key_mat(key, mat, strip_after_chars=strip_after_chars,
                   append_str=append_str, n_dims_keeped=n_dims_keeped)
    key = np.asarray(key)
    save_h5(key, mat, h5file)


# Convert an .ark file to a .h5 file
# The .h5 file should contain 'X' and 'spk_ids' fields
def ark2h5(arkfile, h5_file, n_dims_keeped=None,
           strip_after_chars=None, append_str=None, arktype='vec'):
    key, mat = load_arkfile(arkfile, arktype=arktype)
    key, mat = adjust_key_mat(key, mat, strip_after_chars=strip_after_chars,
                              append_str=append_str, n_dims_keeped=n_dims_keeped)
    key = np.asarray(key)
    save_h5(key, mat, h5_file)


# Convert all .ark files in a given dir to a single .h5 file
# The .h5 file should contain 'X' and 'spk_ids' fields
def arkdir2h5(arkdir, h5_file, pattern='ivector.*.ark', n_dims_keeped=None,
              strip_after_chars=None, append_str=None, arktype='vec'):
    unicode = h5.special_dtype(vlen=str)
    key, mat = load_arkdir(arkdir, pattern=pattern, nfiles=sys.maxsize,
                           arktype=arktype)
    key, mat = adjust_key_mat(key, mat, strip_after_chars=strip_after_chars,
                              append_str=append_str, n_dims_keeped=n_dims_keeped)
    key = np.asarray(key)
    save_h5(key, mat, h5_file)

# Convert an .ark file (must be in 'ary' type because it is a VAD file) 
# to TIMIT .phn file. Assume 8kHz sampling rate and 10ms frame shift.
def ark2phn(arkfile, phndir, arktype='ary', sampling_rate=8000, frame_rate=100):
    fshift_in_sample = int(sampling_rate/frame_rate)
    key, val = load_arkfile(arkfile, arktype=arktype)
    if not os.path.exists(phndir):
        print('Creating ' + phndir)
        os.makedirs(phndir)
    i = 0    
    for k in key:
        phnfile = phndir + '/' + k + '.phn'
        print('Saving %s: %d' % (phnfile, val[i].shape[0]))
        with open(phnfile, 'w') as f:
            time = 0            
            count = 0                           # No. of frames so far from switching point
            t = 1                               # Frame number (start from 0)
            while t < val[i].shape[0]:
                if val[i][t] == val[i][t-1]:
                    count = count + 1
                else:
                    symbol = 'h#' if val[i][t-1] == 0 else 'S'
                    f.write('%s %s %s\n' % (time, time+(count+1)*fshift_in_sample, symbol))
                    time = time + (count+1)*fshift_in_sample      # =80 for 8kHz, 10ms frame shift
                    count = 0
                t = t + 1
            # Last block    
            symbol = 'h#' if val[i][t-1] == 0 else 'S'
            f.write('%s %s %s\n' % (time, time+(count+1)*80, symbol))
        i = i + 1

# Convert all .ark files (indexed by .scp) in a given dir to a single .csv file,
# one row per vector. Key info will be lost.
def scp2csv(basedir, scpfile, csvfile, n_dims_keeped=None, arktype='vec'):
    _, mat = load_scpfile(basedir, scpfile, arktype=arktype)
    if n_dims_keeped is not None and isinstance(mat, np.ndarray):
        mat = mat[:, :n_dims_keeped]
    save_csv(mat, csvfile)


# Convert all .ark files in a given dir to a single .csv file
def arkdir2csv(arkdir, csvfile, pattern='ivector.*.ark', n_dims_keeped=None, arktype='vec'):
    _, mat = load_arkdir(arkdir, pattern=pattern, nfiles=sys.maxsize, arktype=arktype)
    if n_dims_keeped is not None and isinstance(mat, np.ndarray):
        mat = mat[:, :n_dims_keeped]
    save_csv(mat, csvfile)


# Convert an .ark file to a single .csv file
def ark2csv(arkfile, csvfile, n_dims_keeped=None, arktype='vec'):
    _, mat = load_arkfile(arkfile, arktype=arktype)
    if n_dims_keeped is not None and isinstance(mat, np.ndarray):
        mat = mat[:, :n_dims_keeped]
    save_csv(mat, csvfile)


# Convert an .ark file to a Matlab .mat file
def ark2mat(arkfile, matfile, n_dims_keeped=None, arktype='vec'):
    key, mat = load_arkfile(arkfile, arktype=arktype)
    if n_dims_keeped is not None and isinstance(mat, np.ndarray):
        mat = mat[:, :n_dims_keeped]
    if isinstance(mat, np.ndarray):        
        save_mat(key, mat, matfile)
    elif isinstance(mat, list):
        save_lom(key, mat, matfile)
    else:
        raise NotImplementedError


# Convert the .ark file indexex by an .scp file to a Matlab .mat file
def scp2mat(basedir, scpfile, matfile, n_dims_keeped=None, arktype='vec'):
    key, mat = load_scpfile(basedir, scpfile, arktype=arktype)
    if n_dims_keeped is not None:
        mat = mat[:, :n_dims_keeped]
    save_mat(key, mat, matfile)


# Convert .ark or .scp files to .h5, .csv or .mat files
# For ark file type, "vec" means each key is associated with one vector, e.g. i-vec
# "mat" means each key is associated with one matrix, e.g., mfcc
# "ary" means each key is associated with an array; the dim of the arrays
# for different key could be different, e.g. VAD files.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", help="Input filename with extension [ark|scp]")
    parser.add_argument("outfile", help="Output filename with extension [h5|csv|mat]")
    parser.add_argument("basedir", help="Optional base dir of ark files indexed by scp", nargs='?',
                        default=os.getcwd())
    parser.add_argument("-i", "--input_type", help="Input file type [ark|scp], default=ark",
                        default="ark")
    parser.add_argument("-o", "--output_type", help="Output file type [h5|csv|mat], default=h5",
                        default="h5")
    parser.add_argument("-a", "--ark_type", help="Type of ark file [vec|mat|ary|lom], default=vec",
                        default="vec")
    parser.add_argument("-s", "--sampling_rate", help="Sampling rate in Hz",
                        default="8000")
    parser.add_argument("-f", "--frame_rate", help="Frame rate in Hz",
                        default="100")
    args = parser.parse_args()
    if args.input_type == 'ark':
        if args.output_type == 'h5':
            ark2h5(args.infile, args.outfile, arktype=args.ark_type)
        elif args.output_type == 'csv':
            ark2csv(args.infile, args.outfile, arktype=args.ark_type)
        elif args.output_type == 'mat':
            ark2mat(args.infile, args.outfile, arktype=args.ark_type)
        elif args.output_type == 'phn':
            if args.ark_type == 'ary':
                ark2phn(args.infile, args.outfile, arktype=args.ark_type,
                        sampling_rate=float(args.sampling_rate), frame_rate=float(args.frame_rate))
            else:
                print('Ark type must be \'ary\' if the output type is \'phn\'')
        else:
            print('Unsupported output file type %s' % args.output_type)
    elif args.input_type == 'scp':
        if args.output_type == 'h5':
            scp2h5(args.basedir, args.infile, args.outfile, arktype=args.ark_type)
        elif args.output_type == 'csv':
            scp2csv(args.basedir, args.infile, args.outfile, arktype=args.ark_type)
        elif args.output_type == 'mat':
            scp2mat(args.basedir, args.infile, args.outfile, arktype=args.ark_type)
        else:
            print('Unsupported output file type %s' % args.output_type)
    else:
        print('Unsupported input file type %s' % args.input_type)


if __name__ == '__main__':
    main()
