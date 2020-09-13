def parse_list(ndx_lstfile):
    try:
        ndx_lstfile_fh = open(ndx_lstfile)
        temp = ndx_lstfile_fh.read().split()
        spk_logical = [item.split(':')[0] + ':' + item.split('/')[-1] for item in temp]
    finally:
        if ndx_lstfile_fh:
            ndx_lstfile_fh.close()
    return spk_logical