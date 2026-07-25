import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """

    if not seqs:
        return np.array([], dtype=float).reshape(0, 0)
    if not max_len:
        max_len = max(len(s) for s in seqs)
    result = []
    for s in seqs:
        if len(s) < max_len:
            result.append(list(s) + [pad_value] * (max_len - len(s)))
        else:
            result.append(list(s[:max_len]))
    return np.array(result, dtype=float)