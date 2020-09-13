def evl2evl(plda_evlfile, ndxfile, bosaris_evlfile):
    """
     Purpose:
       Convert .evl file produced by score_gplda_w_.py to a format that is acceptable by
       eval_by_bosaris_.py
     Input:
       plda_evlfile    - .evl file produced by score_plda_w_.py
       ndxfile         - .ndx file contains the full path of speech files
     Output:
       bosaris_evlfile - .evl file suitable for Bosaris toolkit
     Format of plda_evlfile:
       100396,tabfsa_sre12.sph,b,1.6729176
     Format of bosaris_evlfile
       110115,.../r141_2_1/sp12-01/data/mic_int/iaaakw-idrzps/iaaeox_sre12.sph,a,-101.6809729
     Format of ndxfile
       110115,.../r141_2_1/sp12-01/data/mic_int/iaaakw-idrzps/iaaeox_sre12.sph,a
     Example usage:
       evl2evl('evl/fw60/gplda60_male_cc1_1024c.evl', '../../ndx/male/core-core_8k_male_cc1.ndx','../../evl/fw60/sre12_gplda60_male_cc1_1024c.evl');

    """

    with open(plda_evlfile, 'r') as fid:
        plda_evlfile_lines = [l.rstrip().split(',') for l in fid]

    with open(ndxfile, 'r') as fid:
        ndxfile_lines = [l.rstrip().split(',') for l in fid]

    plda_evl_dict= {}
    for line in plda_evlfile_lines:
        key = line[0] + '_' + line[1] + '_' + line[2]
        plda_evl_dict.update({key: line[3]})   # line[3]: score

    try:
        fp = open(bosaris_evlfile, 'w')
        for line in ndxfile_lines:
            key = line[0] + '_' + line[1].split('/')[-1] + '_' + line[2]
            fp.write('%s,%s,%s,%s\n' % (line[0], line[1], line[2], plda_evl_dict[key]))
    finally:
        if fp:
            fp.close()
