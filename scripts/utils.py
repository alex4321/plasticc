import io
import os
import pandas as pd
from tqdm import tqdm


class SignalReader:
    def __init__(self, fname):
        self.size = os.path.getsize(fname)
        self.file = open(fname, 'rb')
        self.header = self.file.readline()
        self.positions = {}
        self.next_objects = {}

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

    def object_signal_csv(self, object_id):
        start = self.positions[object_id]
        end = self.positions[self.next_objects[object_id]]
        size = end - start
        self.file.seek(start)
        content = self.file.read(size)
        csv = self.header.strip() + b'\n' + content.strip()
        return csv

    def object_signal(self, object_id):
        container = io.BytesIO()
        container.write(self.object_signal_csv(object_id))
        container.seek(0)
        return pd.read_csv(container)

    def objects_signals(self, objects_ids):
        return pd.concat([self.object_signal(object_id) for object_id in objects_ids],
                         sort=True)