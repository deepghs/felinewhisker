import os.path
import pathlib
from contextlib import contextmanager
from dataclasses import dataclass
from os import PathLike
from typing import Union, Any, Optional, ContextManager, Iterator, Callable

from PIL import Image
from hbutils.random import random_sha1_with_timestamp
from hbutils.system import TemporaryDirectory
from imgutils.data import load_image, grid_transparent


@dataclass
class ImageItem:
    id: str
    image: Union[str, PathLike, Image.Image]
    annotation: Optional[Any]

    def make_pil(self, max_size: int = 1536, background_mode: str = 'white') -> Image.Image:
        if background_mode == 'grid':
            image = load_image(self.image, mode='RGBA', force_background=None)
            image = grid_transparent(image).convert('RGB')
        else:
            image = load_image(self.image, mode='RGB', force_background=background_mode)

        r = (image.width * image.height) / (max_size ** 2)
        if r > 1.0:
            new_width = int(round(image.width / r))
            new_height = int(round(image.height / r))
            image = image.resize((new_width, new_height))

        return image

    @contextmanager
    def make_file(self, max_size: int = 2048, format: str = 'webp', quality: Optional[int] = None,
                  force_reencode: bool = False) -> ContextManager[str]:
        if not force_reencode and isinstance(self.image, (str, PathLike)):
            yield str(pathlib.Path(self.image).resolve())
        else:
            with TemporaryDirectory() as td:
                image = load_image(self.image, mode='RGB', force_background='white')
                r = (image.width * image.height) / (max_size ** 2)
                if r > 1.0:
                    new_width = int(round(image.width / r))
                    new_height = int(round(image.height / r))
                    image = image.resize((new_width, new_height), resample=Image.BICUBIC)

                filename = os.path.join(td, f'{self.id or random_sha1_with_timestamp()}.{format}')
                save_cfg = {}
                if quality:
                    save_cfg['quality'] = quality
                image.save(filename, **save_cfg)

                yield filename


class BaseDataSource:
    def __init__(self, fn_contains_id: Optional[Callable[[str], bool]] = None):
        self._status = 'idle'
        self._fn_contains_id = fn_contains_id or (lambda x: False)

    def set_fn_contains_id(self, fn_contains_id: Optional[Callable[[str], bool]] = None):
        self._fn_contains_id = fn_contains_id or (lambda x: False)

    def _iter(self):
        raise NotImplementedError  # pragma: no cover

    def _init(self):
        pass

    def _close(self):
        pass

    def __init_func(self):
        if self._status == 'idle':
            self._init()
            self._status = 'initialized'
        elif self._status == 'close':
            raise RuntimeError(f'Data source {self!r} already closed, cannot be initialized again.')

    def __close_func(self):
        if self._status == 'idle':
            raise RuntimeError(f'Data source {self!r} not initialized, cannot be closed.')
        elif self._status == 'initialized':
            self._close()

    def close(self):
        self.__close_func()

    def __enter__(self):
        self.__init_func()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__close_func()

    def __iter__(self) -> Iterator[ImageItem]:
        self.__init_func()
        for v in self._iter():
            if isinstance(v, tuple):
                if len(v) == 1:
                    id_ = None
                    v, = v
                    annotate = None
                elif len(v) == 2:
                    id_, v = v
                    annotate = None
                else:
                    id_, v, annotate = v
            else:
                id_ = None
                annotate = None

            id_ = id_ or random_sha1_with_timestamp()
            if not self._fn_contains_id(id_):
                yield ImageItem(id_, v, annotate)
