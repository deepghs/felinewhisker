from typing import Optional, Callable, Union, Iterator

from cheesechaser.datapool import DataPool
from cheesechaser.pipe import Pipe, SimpleImagePipe, PipeItem
from hbutils.string import underscore

from .base import BaseDataSource


class CheeseChaserDataSource(BaseDataSource):
    def __init__(self, source: Union[DataPool, Pipe], id_generator: Iterator[Union[str, int]],
                 source_id: Optional[str] = None, fn_contains_id: Optional[Callable[[str], bool]] = None):
        if isinstance(source, DataPool):
            self._pipe = SimpleImagePipe(source)
            default_source_id = underscore(source.__class__.__name__.replace('DataPool', ''))
        elif isinstance(source, Pipe):
            self._pipe = source
            default_source_id = None
        else:
            raise TypeError(f'Unknown source type - {source!r}.')
        self._source_id = source_id or default_source_id
        self._id_generator = id_generator
        BaseDataSource.__init__(self, fn_contains_id=fn_contains_id)

    def _cid_to_id(self, cid) -> str:
        if self._source_id:
            return f'cheesechaser__{self._source_id}__{cid}'
        else:
            return f'cheesechaser__{cid}'

    def _iter_cids(self):
        for cid in self._id_generator:
            if not self._fn_contains_id(self._cid_to_id(cid)):
                yield cid

    def _iter(self):
        with self._pipe.batch_retrieve(self._iter_cids()) as session:
            for item in session:
                item: PipeItem
                yield self._cid_to_id(item.id), item.data, None
