import argparse
import math
import pandas as pd
import os
import pickle
from tqdm import tqdm
from joblib import Parallel, delayed
from utils import SignalReader


def extract_df_features(pickled_df, fname):
    df = pickle.loads(pickled_df)
    aggs = {
        'mjd': ['min', 'max', 'size'],
        'passband': ['min', 'max', 'mean', 'median', 'std'],
        'flux': ['min', 'max', 'mean', 'median', 'std', 'skew'],
        'flux_err': ['min', 'max', 'mean', 'median', 'std', 'skew'],
        'detected': ['mean'],
        'flux_ratio_sq': ['sum', 'skew'],
        'flux_by_flux_ratio_sq': ['sum', 'skew'],
    }
    agg_train = df.groupby('object_id').agg(aggs)
    new_columns = [
        k + '_' + agg for k in aggs.keys() for agg in aggs[k]
    ]
    agg_train.columns = new_columns
    agg_train['mjd_diff'] = agg_train['mjd_max'] - agg_train['mjd_min']
    agg_train['flux_diff'] = agg_train['flux_max'] - agg_train['flux_min']
    agg_train['flux_dif2'] = (agg_train['flux_max'] - agg_train['flux_min']) / agg_train['flux_mean']
    agg_train['flux_w_mean'] = agg_train['flux_by_flux_ratio_sq_sum'] / agg_train['flux_ratio_sq_sum']
    agg_train['flux_dif3'] = (agg_train['flux_max'] - agg_train['flux_min']) / agg_train['flux_w_mean']

    del agg_train['mjd_max'], agg_train['mjd_min']
    agg_train.head()

    agg_train = agg_train.reset_index()
    agg_train.to_csv(fname, index=None)

    return fname


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--signal_file')
    parser.add_argument('--meta_file')
    parser.add_argument('--window', type=int)
    parser.add_argument('--target_file')
    parser.add_argument('--temporary_directory', default='tmp')
    parser.add_argument('--process_count', type=int)
    parser.add_argument('--batch_size', type=int, default=1000)
    args = parser.parse_args()

    meta = pd.read_csv(args.meta_file)
    object_ids = meta['object_id'].unique()

    if not os.path.exists(args.temporary_directory):
        os.mkdir(args.temporary_directory)

    object_id_batches = []
    object_id_batch_count = int(math.ceil(len(object_ids) / float(args.batch_size)))
    for batch in range(object_id_batch_count):
        batch_ids = object_ids[batch * args.batch_size:][:args.batch_size]
        object_id_batches.append(batch_ids)

    signal_reader = SignalReader(args.signal_file)
    fft_features_files = Parallel(n_jobs=args.process_count)(
        delayed(extract_df_features)(pickle.dumps(signal_reader.objects_signals(objects_ids)),
                                     args.window,
                                     os.path.join(args.temporary_directory,
                                                  'batch-{0}.csv'.format(batch)))
        for batch, objects_ids in enumerate(tqdm(object_id_batches))
    )
    signal_reader.close()

    assert len(fft_features_files) > 0
    features = pd.read_csv(fft_features_files[0])
    for filename in tqdm(fft_features_files[1:]):
        features = pd.concat([features, pd.read_csv(filename)],
                             sort=True)
        os.remove(filename)
    features.to_csv(args.target_file, index=None)
