import os.path
import shutil
import tarfile
import time
from threading import Lock
from typing import Optional, Callable, List, Tuple

import pandas as pd
from PIL import Image
from hbutils.random import random_sha1_with_timestamp
from hbutils.system import TemporaryDirectory
from tqdm import tqdm

from ..tasks import parse_annotation_checker, AnnotationChecker


class RepoAlreadyExistsError(Exception):
    pass


class WriterSession:
    def __init__(self, author: Optional[str], checker: AnnotationChecker,
                 fn_save: Callable[[str, str, str], None], fn_contains_id: Callable[[str], bool]):
        self._author = author
        self._checker = checker
        self.session_token = random_sha1_with_timestamp()
        if self._author:
            self.session_token = f'{self.session_token}__{self._author}'
        self._storage_tmpdir = TemporaryDirectory()
        self._records = {}
        self._fn_save = fn_save
        self._fn_contains_id = fn_contains_id
        self._lock = Lock()

    def is_id_duplicated(self, id_: str) -> bool:
        with self._lock:
            return id_ in self._records or self._fn_contains_id(id_)

    def get_annotated_count(self) -> int:
        count = 0
        for item in self._records.values():
            if item['annotation'] is not None:
                count += 1
        return count

    def add(self, id_: str, image_file: str, annotation):
        with self._lock:
            if annotation is not None:
                self._checker.check(annotation)
            _, ext = os.path.splitext(os.path.basename(image_file))
            filename = f'{id_}{ext}'
            width, height = Image.open(image_file).size
            shutil.copyfile(image_file, os.path.join(self._storage_tmpdir.name, filename))
            self._records[id_] = {
                'id': id_,
                'filename': filename,
                'width': width,
                'height': height,
                'annotation': annotation,
                'updated_at': time.time(),
                'author': self._author,
            }

    def get_image_path(self, id_: str):
        with self._lock:
            return os.path.join(self._storage_tmpdir.name, self._records[id_]['filename'])

    def __getitem__(self, id_):
        with self._lock:
            return self._records[id_]['annotation']

    def __setitem__(self, id_, annotation):
        with self._lock:
            if annotation is not None:
                self._checker.check(annotation)
            self._records[id_]['annotation'] = annotation
            self._records[id_]['updated_at'] = time.time()

    def __delitem__(self, id_):
        with self._lock:
            filename = self._records[id_]['filename']
            del self._records[id_]
            os.remove(os.path.join(self._storage_tmpdir.name, filename))

    def __len__(self):
        with self._lock:
            return len(self._records)

    def __contains__(self, item):
        with self._lock:
            return item in self._records

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
            self._fn_save(tar_file, data_file, self.session_token)

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
        self._exist_ids = None
        self._annotation_checker: Optional[AnnotationChecker] = None
        self._lock = Lock()
        self._sync()

    def _write(self, tar_file: str, data_file: str, token: str):
        raise NotImplementedError  # pragma: no cover

    def _read_meta(self):
        raise NotImplementedError  # pragma: no cover

    def _squash(self):
        raise NotImplementedError  # pragma: no cover

    def _get_table_file(self) -> Optional[str]:
        raise NotImplementedError  # pragma: no cover

    def _list_unarchived_table_files(self) -> List[str]:
        raise NotImplementedError  # pragma: no cover

    def _download_image_file(self, archive_file: str, file_in_archive: str, dst_file: str):
        raise NotImplementedError  # pragma: no cover

    def _exist(self) -> bool:
        raise NotImplementedError  # pragma: no cover

    def _sync(self):
        self.meta_info, self._exist_ids = self._read_meta()
        self._annotation_checker = parse_annotation_checker(self.meta_info)

    def read_table(self) -> Optional[pd.DataFrame]:
        table_file = self._get_table_file()
        if table_file:
            return pd.read_parquet(table_file)
        else:
            return None

    def read_unarchived_tables(self) -> List[Tuple[str, pd.DataFrame]]:
        results = []
        for file in tqdm(self._list_unarchived_table_files()):
            results.append((
                os.path.splitext(os.path.basename(file))[0],
                pd.read_parquet(file)
            ))
        return results

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
                fn_save=self._write,
                fn_contains_id=lambda id_: id_ in self._exist_ids,
            )

    def contains_id(self, id_: str):
        with self._lock:
            return id_ in self._exist_ids

    def is_exist(self):
        with self._lock:
            return self._exist()
