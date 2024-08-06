import logging
import re
from functools import partial
from typing import Optional

import click
from hfutils.utils import ColoredFormatter

from .base import CONTEXT_SETTINGS
from ..repository import HfOnlineRepository, LocalRepository


def _add_init_subcommand(cli: click.Group) -> click.Group:
    @cli.group('init', help='Init dataset.\n\n'
                            'Set environment $HF_TOKEN to use your own access token.',
               context_settings=CONTEXT_SETTINGS)
    def init():
        pass

    @init.command('classification', help='Initialize classification dataset.',
                  context_settings=CONTEXT_SETTINGS)
    @click.option('-d', '--directory', 'directory', type=str, default='.',
                  help='Directory for locally initializing dataset.', show_default=True)
    @click.option('-r', '--repository', 'repository', type=str, default=None,
                  help='Repository for initializing dataset on HuggingFace. '
                       'Will initialize on local directory when not assigned.', show_default=True)
    @click.option('-n', '--name', 'task_name', type=str, required=True,
                  help='Name of this task.')
    @click.option('-l', '--labels', 'labels_str', type=str, required=True,
                  help='Labels for this classification task, seperated with comma.')
    def classification(directory: str, repository: Optional[str], task_name: str, labels_str: str):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(ColoredFormatter())
        logger.addHandler(console_handler)

        if repository:
            logging.info(f'Initializing repository on HuggingFace repo {repository!r} ...')
            fn_init = partial(HfOnlineRepository.init_classification, repo_id=repository)
        else:
            logger.info(f'Initializing repository on local directory {directory!r} ...')
            fn_init = partial(LocalRepository.init_classification, local_dir=directory)

        labels = list(filter(bool, map(str.strip, re.split(r'\s*,\s*', labels_str))))
        logger.info(f'Task name: {task_name!r}, labels: {labels!r}')
        fn_init(
            task_name=task_name,
            labels=labels,
        )

        logger.info('Completed!')

    return cli
