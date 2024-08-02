import glob
import json
import os.path
import shutil
from typing import Optional, List

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.index import tar_get_index_info, tar_file_download
from hfutils.utils import hf_normpath, number_to_tag

from .base import DatasetRepository
from ..tasks import create_readme


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
        return meta_info

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

        meta_file = os.path.join(local_dir, 'meta.json')
        with open(meta_file, 'w') as f:
            json.dump({
                'name': task_name,
                'labels': labels,
                'readme_metadata': readme_metadata,
                'task': 'classification',
            }, f, indent=4, sort_keys=True, ensure_ascii=False),

        md_file = os.path.join(local_dir, 'README.md')
        with open(md_file, 'w') as f:
            readme_metadata['task_categories'] = ['image-classification']
            readme_metadata['size_categories'] = [number_to_tag(0)]
            print(f'---', file=f)
            yaml.dump(readme_metadata, f, default_flow_style=False, sort_keys=False)
            print(f'---', file=f)
            print(f'', file=f)

            print(f'# Image Classification - {task_name}', file=f)
            print(f'', file=f)
            print(f'{plural_word(len(labels), "label")} in total, as the following:', file=f)
            print(f'', file=f)
            for label in labels:
                print(f'* `{label}`', file=f)
            print(f'', file=f)

            print(f'This repository is empty and work in progress currently.', file=f)
            print(f'', file=f)

        return LocalRepository(local_dir)
