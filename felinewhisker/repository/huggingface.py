import json
import logging
import os
import shutil
from typing import Optional, List

import numpy as np
import pandas as pd
from PIL import Image
from hbutils.string import plural_word, humanize
from hbutils.system import TemporaryDirectory
from hfutils.cache import delete_detached_cache
from hfutils.index import tar_get_index_info, hf_tar_file_download
from hfutils.operate import upload_directory_as_directory, get_hf_fs, get_hf_client
from hfutils.utils import hf_normpath, hf_fs_path, parse_hf_fs_path
from huggingface_hub import CommitOperationAdd, CommitOperationDelete
from natsort import natsorted

from .base import DatasetRepository, RepoAlreadyExistsError
from ..tasks import make_readme, init_project


class HfOnlineRepository(DatasetRepository):
    def __init__(self, repo_id: str, revision: str = 'main'):
        self._repo_id = repo_id
        self._revision = revision
        DatasetRepository.__init__(self)

    def _exist(self) -> bool:
        hf_fs = get_hf_fs()
        return hf_fs.exists(hf_fs_path(
            repo_id=self._repo_id,
            repo_type='dataset',
            filename='meta.json',
            revision=self._revision,
        ))

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

            if len(df) > 0:
                named_authors = set(filter(bool, df['author'].tolist()))
            else:
                named_authors = []
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

    def _read_meta(self):
        delete_detached_cache(repo_id=self._repo_id, repo_type='dataset')
        hf_fs = get_hf_fs(hf_token=os.environ.get('HF_TOKEN'))
        hf_client = get_hf_client(hf_token=os.environ['HF_TOKEN'])

        meta_info = json.loads(hf_fs.read_text(hf_fs_path(
            repo_id=self._repo_id,
            repo_type='dataset',
            revision=self._revision,
            filename='meta.json',
        )))
        if hf_fs.exists(hf_fs_path(
                repo_id=self._repo_id,
                repo_type='dataset',
                revision=self._revision,
                filename='data.parquet'
        )):
            df = pd.read_parquet(hf_client.hf_hub_download(
                repo_id=self._repo_id,
                repo_type='dataset',
                revision=self._revision,
                filename='data.parquet'
            ))
            exist_ids = set(df['id'])
        else:
            exist_ids = set()

        return meta_info, exist_ids

    def _get_table_file(self) -> Optional[str]:
        hf_fs = get_hf_fs(hf_token=os.environ.get('HF_TOKEN'))
        hf_client = get_hf_client(hf_token=os.environ['HF_TOKEN'])

        if hf_fs.exists(hf_fs_path(
                repo_id=self._repo_id,
                repo_type='dataset',
                revision=self._revision,
                filename='data.parquet',
        )):
            return hf_client.hf_hub_download(
                repo_id=self._repo_id,
                repo_type='dataset',
                revision=self._revision,
                filename='data.parquet',
            )
        else:
            return None

    def _list_unarchived_table_files(self) -> List[str]:
        hf_fs = get_hf_fs(hf_token=os.environ.get('HF_TOKEN'))
        hf_client = get_hf_client(hf_token=os.environ['HF_TOKEN'])

        files = []
        for filepath in natsorted(hf_fs.glob(hf_fs_path(
                repo_id=self._repo_id,
                repo_type='dataset',
                revision=self._revision,
                filename='unarchived/*.parquet',
        ))):
            filename = parse_hf_fs_path(filepath).filename
            files.append(hf_client.hf_hub_download(
                repo_id=self._repo_id,
                repo_type='dataset',
                revision=self._revision,
                filename=filename,
            ))

        return files

    def _download_image_file(self, archive_file: str, file_in_archive: str, dst_file: str):
        hf_tar_file_download(
            repo_id=self._repo_id,
            repo_type='dataset',
            revision=self._revision,
            archive_in_repo=archive_file,
            file_in_archive=file_in_archive,
            local_file=dst_file,
        )

    def _squash(self):
        delete_detached_cache(repo_id=self._repo_id, repo_type='dataset')
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
        new_authors = set()
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
                if item['author']:
                    new_authors.add(item['author'])
            files_to_drop.append(filename)

        def _load_image_by_id(id_: str):
            selected_item = df[df['id'] == id_].to_dict('records')[0]
            with TemporaryDirectory() as ttd:
                tmp_image_file = os.path.join(
                    ttd, f'image{os.path.splitext(selected_item["filename"])[1]}')
                self._download_image_file(
                    archive_file=selected_item['archive_file'],
                    file_in_archive=selected_item['filename'],
                    dst_file=tmp_image_file,
                )

                image = Image.open(tmp_image_file)
                image.load()
                return image

        with TemporaryDirectory() as td:
            df = pd.DataFrame(list(records.values()))
            if len(df) == 0:
                logging.warning('No samples in total, squash operation cancelled.')
                return
            df = df.sort_values(by=['updated_at', 'id'], ascending=[False, True])
            df.to_parquet(os.path.join(td, 'data.parquet'), engine='pyarrow', index=False)

            make_readme(
                workdir=td,
                task_meta_info=self.meta_info,
                df_samples=df,
                fn_load_image=_load_image_by_id,
            )

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

            commit_message = f'Squash {plural_word(len(files_to_drop), "package")}, ' \
                             f'now this dataset contains {plural_word(len(df), "sample")}'
            if new_authors:
                commit_message = f'{commit_message}, ' \
                                 f'contributed by {", ".join([f"@{name}" for name in sorted(new_authors)])}'

            hf_client.create_commit(
                repo_id=self._repo_id,
                repo_type='dataset',
                revision=self._revision,
                operations=operations,
                commit_message=commit_message,
            )

    def __repr__(self):
        return f'<{self.__class__.__name__} repo_id: {self._repo_id!r}, revision: {self._revision!r}>'

    @classmethod
    def init(cls, task_type: str, repo_id: str, task_name: str,
             readme_metadata: Optional[dict] = None, force: bool = False, **kwargs) -> 'HfOnlineRepository':
        if not force and cls(repo_id=repo_id).is_exist():
            raise RepoAlreadyExistsError(f'HF-based repository {repo_id!r} already exist.')

        hf_client = get_hf_client(hf_token=os.environ.get('HF_TOKEN'))
        readme_metadata = dict(readme_metadata or {})

        with TemporaryDirectory() as td:
            init_project(
                task_type=task_type,
                workdir=td,
                task_name=task_name,
                readme_metadata=readme_metadata,
                **kwargs,
            )

            if not hf_client.repo_exists(repo_id=repo_id, repo_type='dataset'):
                hf_client.create_repo(repo_id=repo_id, repo_type='dataset')

            upload_directory_as_directory(
                repo_id=repo_id,
                repo_type='dataset',
                local_directory=td,
                path_in_repo='.',
                message=f'Initialize {humanize(task_type).lower()} task - {task_name!r}',
                clear=True,
            )

        return HfOnlineRepository(repo_id=repo_id)
