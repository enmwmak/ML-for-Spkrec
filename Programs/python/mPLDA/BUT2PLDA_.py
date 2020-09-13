import numpy as np
# import cupy as np
def BUT2PLDA(spk_logical):
    """
    # Convert BUT's spk_logical to PLDA's spkid format
    """

    _, _, spk_ids = np.unique(spk_logical, return_index=True, return_inverse=True)
    numSpks = len(np.unique(spk_ids))
    numVecs = len(spk_logical)
    PLDA_spkid = np.zeros(shape=(numVecs, numSpks), dtype=np.float64)
    for i in range(numSpks):
        spk_sessions = np.where(spk_ids == i)[0]
        PLDA_spkid[spk_sessions, i] = 1
    return PLDA_spkid