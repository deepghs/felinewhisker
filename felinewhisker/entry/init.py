import io
import logging
from functools import partial
from typing import Optional

import click
import yaml
from hbutils.string import humanize
from hfutils.utils import ColoredFormatter

from .base import CONTEXT_SETTINGS
from ..repository import HfOnlineRepository, LocalRepository
from ..tasks import init_cli, list_task_types


def _make_cmd_for_task_type(task_type: str):
    options, arg_process = init_cli(task_type=task_type)

    @click.option('-d', '--directory', 'directory', type=str, default='.',
                  help='Directory for locally initializing dataset.', show_default=True)
    @click.option('-r', '--repository', 'repository', type=str, default=None,
                  help='Repository for initializing dataset on HuggingFace. '
                       'Will initialize on local directory when not assigned.', show_default=True)
    @click.option('-n', '--name', 'task_name', type=str, required=True,
                  help='Name of this task.')
    def _fn_cli(directory: str, repository: Optional[str], task_name: str, **kwargs):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(ColoredFormatter())
        logger.addHandler(console_handler)

        if repository:
            logging.info(f'Initializing repository on HuggingFace repo {repository!r} ...')
            fn_init = partial(HfOnlineRepository.init, repo_id=repository)
        else:
            logger.info(f'Initializing repository on local directory {directory!r} ...')
            fn_init = partial(LocalRepository.init, local_dir=directory)

        params = dict(
            task_type=task_type,
            task_name=task_name,
            **arg_process(**kwargs),
        )
        with io.StringIO() as sf:
            print('Parameters for initialization:', file=sf)
            yaml.dump(params, sf)
            logger.info(sf.getvalue())
        fn_init(**params)

        logger.info('Completed!')

    for fn_option in options:
        _fn_cli = fn_option(_fn_cli)

    return _fn_cli


def _add_init_subcommand(cli: click.Group) -> click.Group:
    @cli.group('init', help='Init dataset.\n\n'
                            'Set environment $HF_TOKEN to use your own access token.',
               context_settings=CONTEXT_SETTINGS)
    def init():
        pass

    for task_type in list_task_types():
        init.command(
            task_type,
            help=f'Initialize {humanize(task_type).lower()} dataset.',
            context_settings=CONTEXT_SETTINGS
        )(_make_cmd_for_task_type(task_type))

    return cli
