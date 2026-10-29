"""Read only selected frozen Yelp rows; never parse the review JSON for search.

Selection uses the existing pandas sampler on narrow projected metadata, so
timestamp ties, random seeds and frozen-test user grouping remain unchanged.
Full-width Parquet rows are decoded in bounded batches. Metadata is O(N), not
O(small_n); it contains only timestamp/user/label, not text or side tables.
"""
import os

import numpy as np
import pandas as pd


def read_selected_rows(path, positions, batch_size=8192):
    import pyarrow.parquet as pq

    requested = np.asarray(positions, dtype=np.int64)
    positions = np.sort(requested)
    parts = []
    offset = 0
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size):
        end = offset + batch.num_rows
        selected = positions[(positions >= offset) & (positions < end)] - offset
        if len(selected):
            frame = batch.take(selected).to_pandas()
            frame.index = selected + offset
            parts.append(frame)
        offset = end
        if not len(positions) or offset > positions[-1]:
            break
    if not parts:
        return parquet.schema_arrow.empty_table().to_pandas()
    return pd.concat(parts).loc[requested].reset_index(drop=True)


def load_frozen_search_interactions(data, small_n, seed):
    from baselines.common.search_evaluator import _subsample_rec_split_frame
    from baselines.common.budget import memory_checkpoint
    import pyarrow.parquet as pq

    directory = data._std_test_dir(data.data_dir)
    paths = [os.path.join(directory, name) for name in
             ("interaction_frozen.parquet", "std_test.parquet")]
    for path in paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Yelp search requires frozen data: {path}")
    target = data.model_cfg.get("feature", {}).get("target_col", "stars")
    metadata = []
    memory_checkpoint("yelp.before_frozen_metadata")
    for path, split in zip(paths, ("train", "std_test")):
        names = pq.ParquetFile(path).schema_arrow.names
        cols = [c for c in ("user_id", "timestamp", target) if c in names]
        frame = pd.read_parquet(path, columns=list(dict.fromkeys(cols)))
        frame["__source_row__"] = np.arange(len(frame))
        frame["__split__"] = split
        metadata.append(frame)
    selected = _subsample_rec_split_frame(
        pd.concat(metadata, ignore_index=True), data, small_n, seed,
    )
    del metadata
    memory_checkpoint("yelp.sample_selected")
    result = []
    for path, split in zip(paths, ("train", "std_test")):
        positions = selected.loc[selected.__split__ == split, "__source_row__"]
        frame = read_selected_rows(path, positions)
        frame["__split__"] = split
        frame.index = positions.index
        result.append(frame)
    result = pd.concat(result).sort_index().reset_index(drop=True)
    memory_checkpoint("yelp.sample_loaded")
    return result
