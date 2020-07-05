# Perform PLDA scoring using Kaldi's i-vectors
from plda.setup_batch_scoring import extract_group_info, group_enrol_test_ivc
from plda.batch_plda_scoring import plda_scoring
import os
import sys
import time
from plda.h5helper import trial2h5
from kaldi_arkread import scp2h5


# Define constants and options
__extract_group_info = True # Create group info files in data/h5/grouped. Only need to set to True once
__scp2h5 = True             # Read .ark files based onn .scp files
__trial2h5 = True           # Convert trail files to h5 files
vtype = 'ivector'           # Types of vector ('ivectors'|'xvectors')
arktype = 'vec'             # ark files store one vector per key

if vtype == 'ivector':
    basedir = '/home5a/mwmak/so/spkver/sre18-dev/v1-1/'
else:
    basedir = '/home5a/mwmak/so/spkver/sre18-dev/v1-1/'

scpfiles = [
            basedir + 'exp/%ss_sre18_dev_enroll/%s.scp' % (vtype, vtype),
            basedir + 'exp/%ss_sre18_dev_test/%s.scp' % (vtype, vtype)
]
h5dir = 'data/h5/'
h5files = [
            h5dir + '%ss_sre18_dev_enroll.h5' % vtype,
            h5dir + '%ss_sre18_dev_test.h5' % vtype
]
trial_dev = '/corpus/sre18-dev/docs/sre18_dev_trials.tsv'
trial_dev_h5 = 'data/h5/sre18_dev_trials.h5'
grp_info = 'data/lst/group_info_dev.h5'
tgtutt_file = 'data/h5/%ss_sre18_dev_enroll.h5' % vtype
tstutt_file = 'data/h5/%ss_sre18_dev_test.h5' % vtype
tgtutt_grp_file = 'data/h5/grouped/%ss_sre18_dev_enroll.h5' % vtype
tstutt_grp_file = 'data/h5/grouped/%ss_sre18_dev_test.h5' % vtype

# Check input
if len(sys.argv) != 2:
    print('Usage: %s <prep_mode>' % sys.argv[0])
    print('       Valid prep_mode: lda+lennorm')
    exit()

# Convert trial files to .h5 files
if __trial2h5:
    trial_lst = [trial_dev]
    trial_h5_lst = [trial_dev_h5]
    for tsv, h5 in zip(trial_lst, trial_h5_lst):
        os.remove(h5) if os.path.isfile(h5) else None
        print('{0} --> {1}'.format(tsv, h5))
        trial2h5(tsv, h5)


# Read i-vectors from .ark dir and save them as one .h5 file
# Kaldi append '-<segment-name>' to the model id in the enrollment .ark file,
# e.g., 1001_sre18-dlrdnskt_sre18.sph, the segment-name should be removed.
# For the test .ark file, no segment name is appended. As the key does not
# contain the '-' char, it is o.k.
if __scp2h5:
    for adir, hfile in zip(scpfiles, h5files):
        os.remove(hfile) if os.path.isfile(hfile) else None
        print('{0} --> {1}'.format(adir, hfile))
    scp2h5(basedir, scpfiles[0], h5files[0], strip_after_chars=['-'], arktype=arktype)
    scp2h5(basedir, scpfiles[1], h5files[1], arktype=arktype)

# Extract group information in ndx and convert file format
if __extract_group_info:
    filelist = [grp_info, tgtutt_grp_file, tstutt_grp_file]
    for file in filelist:
        os.remove(file) if os.path.isfile(file) else None

    print('{0} --> {1}'.format(trial_dev, grp_info))
    extract_group_info(ndx_file=trial_dev, group_info=grp_info, )

    # group (only 4 group exist in ndx file) enrol and test i-vector
    group_enrol_test_ivc(group_info=grp_info,
                         enrol=tgtutt_file,
                         test=tstutt_file,
                         enrol_grouped=tgtutt_grp_file,
                         test_group=tstutt_grp_file)

# Perform scoring
prep_mode = sys.argv[1]
start_time = time.time()
model_file = h5dir + 'model/kaldi-%s-plda-sre18-%s.h5' % (vtype, prep_mode)
evl_file = os.getcwd() + '/data/evl/kaldi-%s-%s-sre18-eval.evl' % (vtype, prep_mode)
plda_scoring(
    target_file=tgtutt_grp_file,
    test_file=tstutt_grp_file,
    model_file=model_file,
    ndx_file=trial_dev_h5,
    evl_file=evl_file,
)
elapsed_time = time.time() - start_time
print('scoring take {} sec'.format(elapsed_time))

# Compute EER and minDCF
cwd = os.getcwd()
cmd = 'cd ../scoring_software; '
cmd = cmd + 'python3 sre18_submission_scorer.py '
cmd = cmd + '-o %s ' % evl_file
cmd = cmd + '-l /corpus/sre18-dev/docs/sre18_dev_trials.tsv '
cmd = cmd + '-r /corpus/sre18-dev/docs/sre18_dev_trial_key.tsv; '
cmd = cmd + 'cd %s' % cwd
print(cmd)
os.system(cmd)
