import json
import os
import random
import shutil
from typing import Optional, List

import numpy as np
import pandas as pd
import yaml
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.index import tar_get_index_info, hf_tar_file_download
from hfutils.operate import upload_directory_as_directory, get_hf_fs, get_hf_client
from hfutils.utils import hf_normpath, hf_fs_path, parse_hf_fs_path, number_to_tag
from huggingface_hub import CommitOperationAdd, CommitOperationDelete
from natsort import natsorted

from .base import DatasetRepository
from ..utils import padding_align


class HfOnlineRepository(DatasetRepository):
    def __init__(self, repo_id: str, revision: str = 'main'):
        self._repo_id = repo_id
        self._revision = revision
        DatasetRepository.__init__(self)

    def _write(self, tar_file: str, data_file: str, token: str):
        date_str = token[:8]
        with TemporaryDirectory() as td:
            dst_tar_file = os.path.join(td, 'images', date_str, f'{token}.tar')
            dst_idx_file = os.path.join(td, 'images', date_str, f'{token}.json')
            os.makedirs(os.path.dirname(dst_tar_file), exist_ok=True)
            shutil.copyfile(tar_file, dst_tar_file)
            with open(dst_idx_file, 'w') as f:
                json.dump(tar_get_index_info(dst_tar_file, with_hash=True), f)

            dst_data_file = os.path.join(td, 'unarchived', f'{token}.parquet')
            os.makedirs(os.path.dirname(dst_data_file), exist_ok=True)
            df = pd.read_parquet(data_file).replace(np.nan, None)
            records = [
                {**item, 'archive_file': hf_normpath(os.path.relpath(dst_tar_file, td))}
                for item in df.to_dict('records')
            ]
            df = pd.DataFrame(records)
            df.to_parquet(dst_data_file, engine='pyarrow', index=False)

            named_authors = set(filter(bool, df['author'].tolist()))
            pack_name = os.path.basename(dst_tar_file)
            if named_authors:
                commit_message = f'Add package with {plural_word(len(df), "sample")} contributed ' \
                                 f'by {", ".join(map(lambda x: f"@{x}", named_authors))} - {pack_name}'
            else:
                commit_message = f'Add package with {plural_word(len(df), "sample")} - {pack_name}'
            upload_directory_as_directory(
                repo_id=self._repo_id,
                repo_type='dataset',
                revision=self._revision,
                local_directory=td,
                path_in_repo='.',
                message=commit_message,
            )

    def _read(self):
        hf_fs = get_hf_fs(hf_token=os.environ.get('HF_TOKEN'))
        hf_client = get_hf_client(hf_token=os.environ.get('HF_TOKEN'))

        meta_info = json.loads(hf_fs.read_text(hf_fs_path(
            repo_id=self._repo_id,
            repo_type='dataset',
            revision=self._revision,
            filename='meta.json',
        )))
        return meta_info

    def _squash(self):
        hf_fs = get_hf_fs(hf_token=os.environ.get('HF_TOKEN'))
        hf_client = get_hf_client(hf_token=os.environ.get('HF_TOKEN'))

        if hf_fs.exists(hf_fs_path(
                repo_id=self._repo_id,
                repo_type='dataset',
                revision=self._revision,
                filename='data.parquet',
        )):
            df = pd.read_parquet(hf_client.hf_hub_download(
                repo_id=self._repo_id,
                repo_type='dataset',
                revision=self._revision,
                filename='data.parquet',
            ))
            records = {item['id']: item for item in df.to_dict('records')}
        else:
            records = {}

        files_to_drop = []
        for filepath in natsorted(hf_fs.glob(hf_fs_path(
                repo_id=self._repo_id,
                repo_type='dataset',
                revision=self._revision,
                filename='unarchived/*.parquet',
        ))):
            filename = parse_hf_fs_path(filepath).filename
            for item in pd.read_parquet(hf_client.hf_hub_download(
                    repo_id=self._repo_id,
                    repo_type='dataset',
                    revision=self._revision,
                    filename=filename,
            )).to_dict('records'):
                records[item['id']] = item
            files_to_drop.append(filename)

        with TemporaryDirectory() as td:
            df = pd.DataFrame(list(records.values()))
            df = df.sort_values(by=['updated_at', 'id'], ascending=[False, True])
            df.to_parquet(os.path.join(td, 'data.parquet'), engine='pyarrow', index=False)

            md_file = os.path.join(td, 'README.md')
            samples_dir = os.path.join(td, 'samples')
            os.makedirs(samples_dir, exist_ok=True)
            with open(md_file, 'w') as f:
                readme_metadata = self.meta_info['readme_metadata']
                readme_metadata['task_categories'] = ['image-classification']
                readme_metadata['size_categories'] = [number_to_tag(len(df))]
                print(f'---', file=f)
                yaml.dump(readme_metadata, f, default_flow_style=False, sort_keys=False)
                print(f'---', file=f)
                print(f'', file=f)

                print(f'# Image Classification - {self.meta_info["name"]}', file=f)
                print(f'', file=f)
                labels = self.meta_info['labels']
                print(f'{plural_word(len(labels), "label")}, {plural_word(len(df), "sample")} in total, '
                      f'listed as the following:', file=f)
                print(f'', file=f)

                sample_cnt = 8
                samples = []
                for label in labels:
                    df_label = df[df['annotation'] == label]
                    row = {
                        'Label': label,
                        'Samples': f'{len(df_label)} ({len(df_label) / len(df) * 100.0:.1f}%)',
                    }
                    ids = df_label['id'].tolist()
                    if len(ids) > sample_cnt:
                        ids = random.sample(ids, k=sample_cnt)
                    selected = df_label[df_label['id'].isin(ids)].to_dict('records')
                    for i in range(sample_cnt):
                        if i < len(selected):
                            selected_item = selected[i]
                            dst_image_file = os.path.join(samples_dir, label, f'{i}.webp')
                            with TemporaryDirectory() as ttd:
                                tmp_image_file = os.path.join(
                                    ttd, f'image{os.path.splitext(selected_item["filename"])[1]}')
                                hf_tar_file_download(
                                    repo_id=self._repo_id,
                                    repo_type='dataset',
                                    revision=self._revision,
                                    archive_in_repo=selected_item['archive_file'],
                                    file_in_archive=selected_item['filename'],
                                    local_file=tmp_image_file,
                                )
                                image = padding_align(tmp_image_file, (512, 768), color='#00000000')
                                os.makedirs(os.path.dirname(dst_image_file), exist_ok=True)
                                image.save(dst_image_file)

                            row[f'Sample #{i}'] = f'![{label}-{i}]({hf_normpath(os.path.relpath(dst_image_file, td))})'
                        else:
                            row[f'Sample #{i}'] = 'N/A'
                    samples.append(row)
                df_samples = pd.DataFrame(samples)
                print(df_samples.to_markdown(index=False), file=f)
                print(f'', file=f)

            operations = []
            for root, _, files in os.walk(td):
                for file in files:
                    src_file = os.path.abspath(os.path.join(root, file))
                    operations.append(CommitOperationAdd(
                        path_in_repo=hf_normpath(os.path.relpath(src_file, td)),
                        path_or_fileobj=src_file,
                    ))
            for file in files_to_drop:
                operations.append(CommitOperationDelete(path_in_repo=file))

            hf_client.create_commit(
                repo_id=self._repo_id,
                repo_type='dataset',
                revision=self._revision,
                operations=operations,
                commit_message=f'Squash {plural_word(len(files_to_drop), "package")}, '
                               f'now this dataset contains {plural_word(len(df), "sample")}'
            )

    def __repr__(self):
        return f'<{self.__class__.__name__} repo_id: {self._repo_id!r}, revision: {self._revision!r}>'

    @classmethod
    def init_classification(cls, repo_id: str, task_name: str, labels: List[str],
                            readme_metadata: Optional[dict] = None) -> 'HfOnlineRepository':
        hf_client = get_hf_client(hf_token=os.environ.get('HF_TOKEN'))

        from .local import LocalRepository
        with TemporaryDirectory() as td:
            LocalRepository.init_classification(
                local_dir=td,
                task_name=task_name,
                labels=labels,
                readme_metadata=readme_metadata,
            )

            if not hf_client.repo_exists(repo_id=repo_id, repo_type='dataset'):
                hf_client.create_repo(repo_id=repo_id, repo_type='dataset')

            upload_directory_as_directory(
                repo_id=repo_id,
                repo_type='dataset',
                local_directory=td,
                path_in_repo='.',
                message=f'Initialize classification task - {task_name!r}',
                clear=True,
            )

        return HfOnlineRepository(repo_id=repo_id)
