"""Fast sequence loading from the competition parquet files.

Each row group is one 20k-step sequence. Readers keep one ParquetFile per thread
(re-opening parses a large footer) and write straight into a preallocated batch array.
"""
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow.parquet as pq

from utils import FEATURE_COLUMNS, SEQUENCE_LENGTH, TARGET_COLUMNS, WARMUP

N_FEATURES = len(FEATURE_COLUMNS)


class SeqStore:
    def __init__(self, path: str, n_threads: int = 4):
        self.path = path
        pf = pq.ParquetFile(path)
        self.n_seqs = pf.num_row_groups
        self.has_mask = "is_scored" in pf.schema_arrow.names
        self.cols = list(FEATURE_COLUMNS) + list(TARGET_COLUMNS) + (["is_scored"] if self.has_mask else [])
        self._local = threading.local()
        self._pool = ThreadPoolExecutor(n_threads)
        self._cache = None

    def enable_cache(self):
        """Keep every sequence in RAM after its first read (~9 MB each)."""
        self._cache = {}

    def _pf(self):
        if not hasattr(self._local, "pf"):
            self._local.pf = pq.ParquetFile(self.path)
        return self._local.pf

    def _load_into(self, i, g, X, Y, M):
        if self._cache is not None and g in self._cache:
            X[i], Y[i], M[i] = self._cache[g]
            return
        df = self._pf().read_row_group(int(g), columns=self.cols, use_threads=False).to_pandas()
        a = df.to_numpy(np.float32)
        X[i] = a[:, :N_FEATURES]
        Y[i] = a[:, N_FEATURES:N_FEATURES + 2]
        need = np.arange(SEQUENCE_LENGTH) >= WARMUP
        M[i] = need & (a[:, -1] > 0.5) if self.has_mask else need
        if self._cache is not None:
            self._cache[g] = (X[i].copy(), Y[i].copy(), M[i].copy())

    def load(self, ids):
        """-> X [B,T,112] f32, Y [B,T,2] f32, M [B,T] bool (scored rows; all required rows if no mask)."""
        B = len(ids)
        X = np.empty((B, SEQUENCE_LENGTH, N_FEATURES), np.float32)
        Y = np.empty((B, SEQUENCE_LENGTH, 2), np.float32)
        M = np.empty((B, SEQUENCE_LENGTH), bool)
        list(self._pool.map(lambda ig: self._load_into(ig[0], ig[1], X, Y, M), enumerate(ids)))
        return X, Y, M

    def batches(self, ids, batch_size):
        """Yield loaded batches, prefetching the next one in the background."""
        chunks = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]
        if not chunks:
            return
        prefetch = ThreadPoolExecutor(1)
        fut = prefetch.submit(self.load, chunks[0])
        for k in range(len(chunks)):
            hold = [fut.result()]
            fut = prefetch.submit(self.load, chunks[k + 1]) if k + 1 < len(chunks) else None
            yield hold.pop()  # the generator keeps no reference, so the consumer can free the batch
        prefetch.shutdown()


def split_train(n_seqs: int, n_holdout: int, seed: int = 0):
    """Deterministic split of train row groups into (train_ids, holdout_ids)."""
    perm = np.random.default_rng(seed).permutation(n_seqs)
    return np.sort(perm[n_holdout:]), np.sort(perm[:n_holdout])
