import os.path
import shutil
import tarfile
import time
from threading import Lock
from typing import Optional, Callable

import pandas as pd
from hbutils.random import random_sha1_with_timestamp
from hbutils.system import TemporaryDirectory
from tqdm import tqdm

from ..tasks import parse_annotation_checker_from_meta, AnnotationChecker


class WriterSession:
    def __init__(self, author: Optional[str], checker: AnnotationChecker,
                 save_func: Callable[[str, str, str], None]):
        self._author = author
        self._checker = checker
        self._token = random_sha1_with_timestamp()
        if self._author:
            self._token = f'{self._token}__{self._author}'
        self._storage_tmpdir = TemporaryDirectory()
        self._records = {}
        self._save_func = save_func
        self._lock = Lock()

    def add(self, id_: str, image_file: str, annotation):
        with self._lock:
            if annotation is not None:
                self._checker.check(annotation)
            _, ext = os.path.splitext(os.path.basename(image_file))
            filename = f'{id_}{ext}'
            shutil.copyfile(image_file, os.path.join(self._storage_tmpdir.name, filename))
            self._records[id_] = {
                'id': id_,
                'filename': filename,
                'annotation': annotation,
                'updated_at': time.time(),
                'author': self._author,
            }

    def __getitem__(self, item):
        with self._lock:
            return self._records[item]['annotation']

    def __setitem__(self, key, value):
        with self._lock:
            if value is not None:
                self._checker.check(value)
            self._records[key]['annotation'] = value
            self._records[key]['updated_at'] = time.time()

    def __delitem__(self, key):
        with self._lock:
            filename = self._records[key]['filename']
            del self._records[key]
            os.remove(os.path.join(self._storage_tmpdir.name, filename))

    def __len__(self):
        with self._lock:
            return len(self._records)

    def _save(self):
        with TemporaryDirectory() as td:
            records = []
            tar_file = os.path.join(td, 'data.tar')
            with tarfile.open(tar_file, 'a:') as tar:
                keys = sorted(self._records.keys())
                for key in tqdm(keys, desc='Packing'):
                    item = self._records[key]
                    filename = item['filename']
                    if item['annotation'] is not None:
                        tar.add(os.path.join(self._storage_tmpdir.name, filename), filename)
                        records.append(item)

            data_file = os.path.join(td, 'data.parquet')
            df = pd.DataFrame(records)
            df.to_parquet(data_file, engine='pyarrow', index=False)
            self._save_func(tar_file, data_file, self._token)

    def save(self):
        with self._lock:
            self._save()

    def _close(self):
        self._storage_tmpdir.cleanup()

    def close(self):
        with self._lock:
            self._close()

    def __del__(self):
        self._close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._save()
        self._close()


class DatasetRepository:
    def __init__(self):
        self.meta_info = None
        self._annotation_checker: Optional[AnnotationChecker] = None
        self._lock = Lock()
        self._sync()

    def _write(self, tar_file: str, data_file: str, token: str):
        raise NotImplementedError  # pragma: no cover

    def _read(self):
        raise NotImplementedError  # pragma: no cover

    def _squash(self):
        raise NotImplementedError  # pragma: no cover

    def _sync(self):
        self.meta_info = self._read()
        self._annotation_checker = parse_annotation_checker_from_meta(self.meta_info)

    def squash(self):
        with self._lock:
            self._sync()
            self._squash()
            self._sync()

    def sync(self):
        with self._lock:
            self._sync()

    def write(self, author: Optional[str] = None):
        with self._lock:
            return WriterSession(
                author=author,
                checker=self._annotation_checker,
                save_func=self._write,
            )
