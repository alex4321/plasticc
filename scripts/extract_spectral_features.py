import argparse
import math
import io
import pickle
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from utils import SignalReader


def rolling(signal, window):
    if len(signal) < window:
        return np.ones([1, window]) * signal.mean()
    shape = (signal.shape[0] - window + 1, window)
    strides = (signal.itemsize, signal.itemsize)
    return np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)


def series_rolling_fft(signal, window):
    signal_rolling = rolling(signal.values, window)
    fft_values = np.fft.fft(signal_rolling, axis=1)[:, :window//2] / window
    fft_amplitudes = np.abs(fft_values)
    return np.median(fft_amplitudes, axis=0)


def rolling_fft_df(df, flux_column, window):
    spectrums_series = df.groupby(['object_id', 'passband'])[flux_column].apply(
        lambda flux: series_rolling_fft(flux, window)
    ).reset_index()
    spectrums_series['flux_mean'] = spectrums_series[flux_column].apply(np.mean)
    spectrums_series['flux_std'] = spectrums_series[flux_column].apply(np.std)
    spectrums_series[flux_column] = spectrums_series.apply(
        lambda row: (row[flux_column] - row['flux_mean']) / (row['flux_std'] + 1e-10),
        axis=1
    )
    spectrums = np.array(list(
        spectrums_series[flux_column]
    ))
    flux_columns = [
        flux_column + '_freq_{0}'.format(i)
        for i in range(spectrums.shape[1])
    ]
    object_channel_columns = ['object_id', 'passband', 'flux_mean', 'flux_std']
    spectrums_df = pd.DataFrame(spectrums, columns=flux_columns)
    for column in object_channel_columns:
        spectrums_df[column] = spectrums_series[column]
    return spectrums_df


def passband_fft_features(spectrum_features):
    data = None
    flux_columns = [column
                    for column in spectrum_features.columns
                    if column.startswith('flux_')]
    for passband in range(6):
        passband_data = spectrum_features.loc[spectrum_features['passband'] == passband].copy(deep=False)
        rename = {
            column: 'passband_{0}_{1}'.format(passband, column)
            for column in flux_columns
        }
        passband_data.rename(columns=rename, inplace=True)
        del passband_data['passband']
        if data is None:
            data = passband_data
        else:
            data = data.merge(passband_data, left_on='object_id', right_on='object_id')
    return data


def extract_df_features(pickled_df, window, fname):
    df = pickle.loads(pickled_df)
    features = passband_fft_features(rolling_fft_df(df, 'flux', window))
    features.to_csv(fname, index=None)
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
