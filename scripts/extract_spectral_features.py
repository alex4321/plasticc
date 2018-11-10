import argparse
import math
import io
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed


class SignalReader:
    def __init__(self, fname, buffersize=5):
        self.size = os.path.getsize(fname)
        self.file = open(fname, 'rb')
        self.header = self.file.readline()
        self.positions = {}
        self.next_objects = {}
        self.buffersize = buffersize
        self.buffer = {}

        previous_object = None
        line_num = 1
        with tqdm(total=self.size // (1024 * 1024)) as pbar:
            while True:
                position = self.file.tell()
                info = self.file.read(1024 * 1024)
                if not info:
                    break
                pbar.update(1)
                lines = info.splitlines(True)
                offset = 0
                for line in lines:
                    line_num += 1
                    if line:
                        if line.endswith(b'\n'):
                            object_id = int(line.split(b',', 1)[0])
                            if object_id != previous_object:
                                if previous_object is not None:
                                    self.next_objects[previous_object] = object_id
                                self.positions[object_id] = position + offset
                                previous_object = object_id
                            offset += len(line)
                        else:
                            self.file.seek(position + offset)
        self.next_objects[previous_object] = 'END'
        self.positions['END'] = self.file.tell()

    def close(self):
        self.file.close()

    def object_signal_csv(self, object_id, bufferize_next=None):
        if bufferize_next is None:
            if len(self.buffer) > self.buffersize:
                self.buffer = {}
            bufferize_next = self.buffersize
        if object_id not in self.buffer:
            start = self.positions[object_id]
            end = self.positions[self.next_objects[object_id]]
            size = end - start
            self.file.seek(start)
            content = self.file.read(size)
            csv = self.header.strip() + b'\n' + content.strip()
            self.buffer[object_id] = csv
        if bufferize_next > 0:
            next_id = self.next_objects[object_id]
            if next_id != 'END':
                self.object_signal_csv(next_id, bufferize_next - 1)
        return self.buffer[object_id]

    def object_signal(self, object_id):
        container = io.BytesIO()
        container.write(self.object_signal_csv(object_id))
        container.seek(0)
        return pd.read_csv(container)

    def objects_signals(self, objects_ids):
        return pd.concat([self.object_signal(object_id) for object_id in object_ids],
                         sort=True)


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


def extract_df_features(df, window, fname):
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

    signal_reader = SignalReader(args.signal_file, args.batch_size)
    fft_features_files = Parallel(n_jobs=args.process_count)(
        delayed(extract_df_features)(signal_reader.objects_signals(objects_ids),
                                     args.window,
                                     os.path.join(args.temporary_directory, 'batch-{0}.csv'))
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

