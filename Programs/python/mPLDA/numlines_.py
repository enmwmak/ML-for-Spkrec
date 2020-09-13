def numlines(lst_file_name):
    try:
        lst_file_fh= open(lst_file_name)
        nlines = len(lst_file_fh.read().split())
    finally:
        if lst_file_fh:
            lst_file_fh.close()
    return nlines