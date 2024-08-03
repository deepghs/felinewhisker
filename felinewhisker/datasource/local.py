import mimetypes
import os.path
import re
from typing import Optional, Callable

from .base import BaseDataSource

mimetypes.add_type('image/webp', '.webp')


class LocalDataSource(BaseDataSource):
    def __init__(self, local_dir: str, source_id: Optional[str] = None,
                 fn_contains_id: Optional[Callable[[str], bool]] = None):
        BaseDataSource.__init__(self, fn_contains_id=fn_contains_id)
        self.local_dir = os.path.abspath(os.path.normpath(os.path.expanduser(os.path.normcase(local_dir))))
        self.source_id = source_id or re.sub(r'[\W_]+', '_', self.local_dir).strip('_')

    def _iter(self):
        for root, _, files in os.walk(self.local_dir):
            for file in files:
                path = os.path.abspath(os.path.join(root, file))
                mimetype, _ = mimetypes.guess_type(file)
                if mimetype.startswith('image/'):
                    file_token = re.sub(r'[\W_]+', '_', os.path.relpath(path, self.local_dir)).strip('_')
                    id_ = f'localdir__{self.source_id}__{file_token}'
                    if not self._fn_contains_id(id_):
                        yield id_, path, None
