import re
from typing import Optional, Union, List

from prompt_toolkit.document import Document
from prompt_toolkit.validation import Validator, ValidationError


class HuggingFaceRepoValidator(Validator):
    __PATTERN__ = re.compile(r'^([a-z0-9][a-z0-9_-]*)/([a-z0-9](?:[a-z0-9_-]{0,94}[a-z0-9])?)$')

    def __init__(self, message: str = 'Invalid HuggingFace repository name.'):
        self._message = message

    def validate(self, document: Document) -> None:
        if not self.__PATTERN__.fullmatch(document.text):
            raise ValidationError(message=self._message)


class StringNonEmptyValidator(Validator):
    def __init__(self, min_length: int = 1, max_length: Optional[int] = None):
        self._min_length = min_length
        self._max_length = max_length

    def validate(self, document: Document) -> None:
        text = document.text.strip()
        if len(text) < self._min_length:
            raise ValidationError(message=f'Length should be no less than {self._min_length}, '
                                          f'but {len(text)} actually given.')

        if self._max_length is not None and len(text) > self._max_length:
            raise ValidationError(message=f'Length should be no more than {self._max_length}, '
                                          f'but {len(text)} actually given.')


class MultiStringEmptyValidator(Validator):
    def __init__(self, splitter: Union[re.Pattern, str] = ',',
                 min_length: int = 1, max_length: Optional[int] = None,
                 min_count: int = 1, max_count: Optional[int] = None, allow_duplicate: bool = False):
        self._splitter = splitter
        self._min_length = min_length
        self._max_length = max_length
        self._min_count = min_count
        self._max_count = max_count
        self._allow_duplicate = allow_duplicate

    def _split_to_labels(self, text: str) -> List[str]:
        if isinstance(self._splitter, str):
            segments = text.split(self._splitter)
        elif isinstance(self._splitter, re.Pattern):
            segments = self._splitter.split(text)
        else:
            raise TypeError(f'Invalid splitter - {self._splitter!r}.')
        return list(map(str.strip, segments))

    def validate(self, document: Document) -> None:
        segments = self._split_to_labels(document.text)

        if len(segments) < self._min_count:
            raise ValidationError(
                message=f'Count of items should be no less than {self._min_count}, '
                        f'but {len(segments)} actually given.'
            )
        if self._max_count is not None and len(segments) > self._max_count:
            raise ValidationError(
                message=f'Count of items should be no more than {self._max_count}, '
                        f'but {len(segments)} actually given.'
            )

        _exist_items = set()
        for item in segments:
            if len(item) < self._min_length:
                raise ValidationError(
                    message=f'Item length should be no less than {self._min_length}, '
                            f'but {len(item)} actually given - {item!r}.'
                )
            if self._max_length is not None and len(item) > self._max_length:
                raise ValidationError(
                    message=f'Item length should be no more than {self._max_length}, '
                            f'but {len(item)} actually given - {item!r}.'
                )
            if not self._allow_duplicate and item in _exist_items:
                raise ValidationError(
                    message=f'Duplicated item found - {item!r}.'
                )
            _exist_items.add(item)




