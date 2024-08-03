import glob
import json
import logging
import os.path
import shutil
from typing import Optional, List

import numpy as np
import pandas as pd
from PIL import Image
from hbutils.system import TemporaryDirectory
from hfutils.index import tar_get_index_info, tar_file_download
from hfutils.utils import hf_normpath

from .base import DatasetRepository
from ..tasks import create_readme, init_project


class LocalRepository(DatasetRepository):
    def __init__(self, repo_dir: str):
        self._repo_dir = repo_dir
        self._meta_info_file = os.path.join(self._repo_dir, 'meta.json')
        self._data_file = os.path.join(self._repo_dir, 'data.parquet')
        DatasetRepository.__init__(self)

    def _write(self, tar_file: str, data_file: str, token: str):
        date_str = token[:8]
        dst_tar_file = os.path.join(self._repo_dir, 'images', date_str, f'{token}.tar')
        dst_idx_file = os.path.join(self._repo_dir, 'images', date_str, f'{token}.json')
        os.makedirs(os.path.dirname(dst_tar_file), exist_ok=True)
        shutil.copyfile(tar_file, dst_tar_file)
        with open(dst_idx_file, 'w') as f:
            json.dump(tar_get_index_info(dst_tar_file, with_hash=True), f)

        dst_data_file = os.path.join(self._repo_dir, 'unarchived', f'{token}.parquet')
        os.makedirs(os.path.dirname(dst_data_file), exist_ok=True)
        df = pd.read_parquet(data_file).replace(np.nan, None)
        records = [
            {**item, 'archive_file': hf_normpath(os.path.relpath(dst_tar_file, self._repo_dir))}
            for item in df.to_dict('records')
        ]
        df = pd.DataFrame(records)
        df.to_parquet(dst_data_file, engine='pyarrow', index=False)

    def _read(self):
        with open(self._meta_info_file, 'r') as f:
            meta_info = json.load(f)
        if os.path.exists(self._data_file):
            exist_ids = set(pd.read_parquet(self._data_file)['id'])
        else:
            exist_ids = set()
        return meta_info, exist_ids

    def _squash(self):
        data_file = os.path.join(self._repo_dir, 'data.parquet')
        if os.path.exists(data_file):
            df = pd.read_parquet(data_file)
            records = {item['id']: item for item in df.to_dict('records')}
        else:
            records = {}

        files_to_drop = []
        for file in glob.glob(os.path.join(self._repo_dir, 'unarchived', '*.parquet')):
            for item in pd.read_parquet(file).to_dict('records'):
                records[item['id']] = item
            files_to_drop.append(file)
        df = pd.DataFrame(list(records.values()))
        if len(df) == 0:
            logging.warning('No samples in total, squash operation cancelled.')
        df = df.sort_values(by=['updated_at', 'id'], ascending=[False, True])
        df.to_parquet(data_file, engine='pyarrow', index=False)
        for file in files_to_drop:
            os.remove(file)

        def _load_image_by_id(id_: str):
            selected_item = df[df['id'] == id_].to_dict('records')[0]
            with TemporaryDirectory() as ttd:
                tmp_image_file = os.path.join(
                    ttd, f'image{os.path.splitext(selected_item["filename"])[1]}')
                tar_file_download(
                    archive_file=os.path.join(self._repo_dir, selected_item['archive_file']),
                    file_in_archive=selected_item['filename'],
                    local_file=tmp_image_file,
                )

                image = Image.open(tmp_image_file)
                image.load()
                return image

        md_file = os.path.join(self._repo_dir, 'README.md')
        with open(md_file, 'w') as f:
            create_readme(
                f=f,
                workdir=self._repo_dir,
                task_meta_info=self.meta_info,
                df_samples=df,
                fn_load_image=_load_image_by_id,
            )

    def __repr__(self):
        return f'<{self.__class__.__name__} dir: {self._repo_dir!r}>'

    @classmethod
    def init_classification(cls, local_dir: str, task_name: str, labels: List[str],
                            readme_metadata: Optional[dict] = None) -> 'LocalRepository':
        os.makedirs(local_dir, exist_ok=True)
        readme_metadata = dict(readme_metadata or {})
        init_project(
            task_type='classification',
            workdir=local_dir,
            task_name=task_name,
            readme_metadata=readme_metadata,
            labels=labels,
        )

        return LocalRepository(local_dir)
