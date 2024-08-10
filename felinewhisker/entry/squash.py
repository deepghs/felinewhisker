import logging
from typing import Optional

import click
from hfutils.utils import get_requests_session, ColoredFormatter
from huggingface_hub import configure_http_backend

from .base import CONTEXT_SETTINGS, ClickErrorException
from ..repository import LocalRepository, HfOnlineRepository


class NoDatasetAssigned(ClickErrorException):
    exit_code = 0x10


def _add_squash_subcommand(cli: click.Group) -> click.Group:
    @cli.command('squash', help='Squash a dataset by merging all the unarchived contributions.\n\n'
                                'Set environment $HF_TOKEN to use your own access token.',
                 context_settings=CONTEXT_SETTINGS)
    @click.option('-d', '--directory', 'directory', type=str, default=None,
                  help='Local directory of the dataset.', show_default=False)
    @click.option('-r', '--repository', 'repository', type=str, default=None,
                  help='HuggingFace Repository of the dataset.', show_default=False)
    def squash(directory: Optional[str], repository: Optional[str]):
        configure_http_backend(get_requests_session)

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(ColoredFormatter())
        logger.addHandler(console_handler)

        if repository:
            repo = HfOnlineRepository(repo_id=repository)
        elif directory:
            repo = LocalRepository(directory)
        else:
            raise NoDatasetAssigned(
                'No dataset assigned. '
                'You have to use either -d or -r option to assign a local or a HF-based dataset.'
            )

        repo.squash()

    return cli
