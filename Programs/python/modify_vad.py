"""
# Read an ark file and an SRE18 diarization file.
# Modify the ark file based on the diarization file without
# changing the number of frames and file size of the input
# ark file. Then, save the modified VAD info to an output ark file.
#
Example:
python3 modify_vad.py /corpus/sre18-dev/docs/sre18_dev_enrollment_diarization.tsv \
    /home12a/mwmak12/sre18-dev/v2-1/mfcc/vad_sre18_dev_enroll.8.ark \
    /home12a/mwmak12/sre18-dev/v2-1/vad_tmp/vad_sre18_dev_enroll.8.ark    
"""

import kaldi_io
import numpy as np
import argparse
from kaldi_arkread import load_ary_arkfile


def get_diar_info(lines, frame_rate=100):
    diar_info = {}
    last_turn_endframe = int(0 * frame_rate)
    for line in lines:
        field = line.split('\t')
        segmentid = field[0]
        start_frame = int(np.ceil(float(field[2]) * frame_rate))
        end_frame = int(np.ceil(float(field[3]) * frame_rate))
        if segmentid not in diar_info:
            diar_info[segmentid] = [0.0] * (start_frame - 1)
            last_turn_endframe = end_frame
        else:
            diar_info[segmentid] = diar_info[segmentid] + \
                    [0.0]*(start_frame-last_turn_endframe) # Bug fixed on 15 Aug.  
            last_turn_endframe = end_frame
        diar_info[segmentid] = diar_info[segmentid] + \
                    [1.0]*(end_frame-start_frame) # Bug fixed on 15 Aug.
        #print('%s: %d %d %d %d %d' % (segmentid, start_frame, end_frame, diar_info[segmentid].count(0), diar_info[segmentid].count(1), len(diar_info[segmentid])))
    return diar_info


# Convert SRE18 diarization file to vad file (.ark)
# The diarization file is a .tsv file containing the following fields
# segmentid       speaker_type  start     end
# ogwhdmry_sre18.flac     target  0.09    7.31
# Diarization file: /corpus/sre18-dev/docs/sre18_dev_enrollment_diarization.tsv
# Note that the default frame_shift in Kaldi is 10ms
def modify_vad(diarfile, in_vadfile, out_vadfile, frame_shift=10):

    # Get diarization info from diarization file
    print('Reading %s' % diarfile)
    with open(diarfile, 'rt') as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    lines.pop(0)                        # Remove header
    frame_rate = 1/(frame_shift/1000)   # Frame rate in Hz (default is 100Hz)
    diar_info = get_diar_info(lines, frame_rate=frame_rate)

    # Get vad info from vad file (in ark format) created by compute-vad
    print('Reading %s' % in_vadfile)
    vadkey, vad = load_ary_arkfile(in_vadfile)
    vad_info = dict(zip(vadkey, vad))

    # Perform elementwise logical AND between vad info from diar_info. As VAD info is either
    # 0 or 1, we use np.multiply. This ensures that the results have the same type as VAD info
    # Starting from Numpy 1.16, we cannot set the flag write=True. 
    for vadkey in vad_info.keys(): 
        diarkey = vadkey.split('-', 1)[-1]   # Use the filename after '-' if '-' exists
        if diarkey in diar_info:             # Only .flac files need diarization
            n_copy = int(np.min([len(vad_info[vadkey]), len(diar_info[diarkey])]))
            #vad_info[vadkey].setflags(write=True)
            new_val = np.multiply(vad_info[vadkey][0:n_copy], diar_info[diarkey][0:n_copy])
            new_val = np.array(new_val, dtype='float32')
            tmp_arr = np.copy(vad_info[vadkey])             # Create a copy of the array in python process heap 
            np.put(tmp_arr, range(n_copy), new_val)         # Modify the first n_copy elements in tmp_arr
            vad_info[vadkey] = tmp_arr

    # Write diar_info to .ark file
    print('Writing %s' % out_vadfile)
    with open(out_vadfile, 'wb') as f:
        for key in vad_info.keys():
            val = np.asarray(vad_info[key], dtype='float32')
            kaldi_io.write_vec_flt(f, val, key=key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("diafile",
                        help=("Input diarzation file, e.g., "
                              "/corpus/sre18-dev/docs/sre18_dev_enrollment_diarization.tsv"))
    parser.add_argument("in_vadfile",
                        help=("Input VAD file in .ark format, e.g., "
                              "/home5a/mwmak/so/spkver/sre18-dev/v1-1/mfcc/vad_sre18_dev_enroll.16.ark"))
    parser.add_argument("out_vadfile", help="Output VAD file in .ark format")
    args = parser.parse_args()
    modify_vad(args.diafile, args.in_vadfile, args.out_vadfile)


if __name__ == '__main__':
    main()
