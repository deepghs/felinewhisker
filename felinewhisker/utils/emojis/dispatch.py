import logging
import pathlib
import re
from functools import lru_cache
from urllib.parse import quote_plus

from emoji import emojize
from hfutils.utils import download_file
from pilmoji.source import EmojiCDNSource

_EMOJIS_DIR = pathlib.Path(__file__).parent


@lru_cache()
def emoji_image_file(emoji: str, style: str = 'twitter') -> str:
    short_tag = re.sub(r'[\W_]+', '_', emoji).strip('_').strip()
    expected_file_path = _EMOJIS_DIR / f'{short_tag}.png'

    if not expected_file_path.exists():
        class _CustomSource(EmojiCDNSource):
            STYLE = style

            def get_url(self, emoji: str, /) -> str:
                return self.BASE_EMOJI_CDN_URL + quote_plus(emoji) + '?style=' + quote_plus(self.STYLE)

        url = _CustomSource().get_url(emojize(emoji))
        logging.info(f'Downloading {url!r} to {str(expected_file_path)!r} ...')
        download_file(
            url=url,
            filename=str(expected_file_path),
        )

    return str(expected_file_path)
