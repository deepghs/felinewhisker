import os.path
import tarfile
import time
import warnings
from threading import Lock
from typing import Union, Optional

import pandas as pd
from hbutils.scale import size_to_bytes
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory

from ..tasks import parse_annotation_checker_from_meta, AnnotationChecker


class RepoWriter:
    def __init__(self):
        self._tmpdir = TemporaryDirectory()
        self._tar_file = os.path.join(self._tmpdir.name, 'storage.tar')
        self._data_file = os.path.join(self._tmpdir.name, 'data.parquet')
        self._tar = tarfile.open(self._tar_file, mode='a:')
        self._records = []
        self._file_count = 0
        self._file_total_size = 0

    @property
    def file_count(self) -> int:
        return self._file_count

    @property
    def file_total_size(self) -> int:
        return self._file_total_size

    def add_file(self, id_: Union[str, int], image_file: str, annotation, author: Optional[str] = None):
        _, ext = os.path.splitext(os.path.basename(image_file))
        filename = f'{id_}{ext}'
        self._tar.add(image_file, filename)
        self._records.append({
            'id': id_,
            'filename': filename,
            'annotation': annotation,
            'created_at': time.time(),
            'author': author,
        })
        self._file_count += 1
        self._file_total_size += os.path.getsize(image_file)

    def flush(self):
        df = pd.DataFrame(self._records)
        df.to_parquet(self._data_file, engine='pyarrow', index=False)
        self._tar.close()
        self._tar = tarfile.open(self._tar_file, mode='a:')
        return self._tar_file, self._data_file

    def close(self):
        self._tmpdir.cleanup()


class DatasetRepository:
    def __init__(self, flush_files: Optional[int] = None, flush_size: Optional[Union[int, str]] = None):
        self._flush_files = flush_files
        self._flush_size = flush_size
        if self._flush_size:
            self._flush_size = int(size_to_bytes(self._flush_size))
        self._writer: Optional[RepoWriter] = None
        self.meta_info, self._exist_ids = None, None
        self._annotation_checker: Optional[AnnotationChecker] = None
        self._lock = Lock()
        self._sync()

    def _write(self, tar_file, data_file):
        raise NotImplementedError  # pragma: no cover

    def _read(self):
        raise NotImplementedError  # pragma: no cover

    def _squash(self):
        raise NotImplementedError  # pragma: no cover

    def _sync(self):
        if self._writer and self._writer.file_count > 0:
            warnings.warn(f'Unsaved {plural_word(self._writer.file_count, "image")} will be dropped when syncing.')
        self._writer = None
        self.meta_info, self._exist_ids = self._read()
        self._annotation_checker = parse_annotation_checker_from_meta(self.meta_info)

    def squash(self):
        with self._lock:
            self._sync()
            self._squash()
            self._sync()

    def sync(self):
        with self._lock:
            self._sync()

    def add_file(self, id_: Union[str, int], image_file: str, annotation, author: Optional[str] = None):
        with self._lock:
            if id_ in self._exist_ids:
                warnings.warn(f'Id already exist in {self}, skipped.')
                return
            if self._annotation_checker:  # check the annotation
                self._annotation_checker.check(annotation)

            if self._writer is None:
                self._writer = RepoWriter()
            self._writer.add_file(id_=id_, image_file=image_file, annotation=annotation, author=author)
            self._exist_ids.add(id_)
            if (self._flush_files is not None and self._writer.file_count >= self._flush_files) or \
                    (self._flush_size is not None and self._writer.file_total_size >= self._flush_size):
                self._flush()

    def _flush(self):
        if self._writer is not None:
            tar_file, data_file = self._writer.flush()
            self._write(tar_file=tar_file, data_file=data_file)
            self._writer.close()
            self._writer = None

    def flush(self):
        with self._lock:
            self._flush()
