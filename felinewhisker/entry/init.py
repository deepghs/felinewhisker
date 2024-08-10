import copy
import os
from functools import partial
from pathlib import Path

import click
from InquirerPy import inquirer
from hbutils.collection import unique
from hbutils.string import titleize
from hfutils.utils import get_requests_session
from huggingface_hub import configure_http_backend
from prompt_toolkit.document import Document
from prompt_toolkit.validation import Validator, ValidationError

from .base import CONTEXT_SETTINGS
from ..repository import LocalRepository, HfOnlineRepository, RepoAlreadyExistsError
from ..tasks import init_cli, list_task_types
from ..utils import HuggingFaceRepoValidator, StringNonEmptyValidator, hf_licence


class NewDirectoryValidator(Validator):
    def __init__(self, message: str = "Input is not a valid directory"):
        self._message = message

    def validate(self, document: Document) -> None:
        path = Path(document.text).expanduser()
        if path.exists() and not path.is_dir():
            raise ValidationError(message=self._message)


def _add_init_subcommand(cli: click.Group) -> click.Group:
    @cli.command('init', help='Init a dataset.\n\n'
                              'Set environment $HF_TOKEN to use your own access token.',
                 context_settings=CONTEXT_SETTINGS)
    def init():
        configure_http_backend(get_requests_session)

        while True:
            init_position = inquirer.select(
                message="Where are you planning to initialize this dataset?",
                choices=[
                    {"name": "On My Local Directory", "value": "local"},
                    {"name": "On HuggingFace Repository (Recommended)", "value": "huggingface"},
                ],
                default="huggingface",
            ).execute()

            if init_position == 'huggingface' or inquirer.confirm(
                    message='We strongly recommend you to use HuggingFace Repository to maintain your dataset '
                            'annotation working progress.\n'
                            'It provide many useful features like git-based data managements.\n'
                            'If you use local directory these will all be unavailable. Are you sure?',
            ).execute():
                break

        if init_position == 'local':
            init_directory = inquirer.filepath(
                message="Which directory to initialize the dataset?",
                validate=NewDirectoryValidator(message="Please enter a valid directory!"),
                only_directories=True,
                default=os.getcwd(),
            ).execute()
            fn_init = partial(LocalRepository.init, local_dir=init_directory)
        elif init_position == 'huggingface':
            init_repo = inquirer.text(
                message="Which HF repository to initialize the dataset?",
                validate=HuggingFaceRepoValidator(),
            ).execute()
            fn_init = partial(HfOnlineRepository.init, repo_id=init_repo)
        else:
            raise ValueError(f'Unknown init position - {init_position!r}.')  # pragma: no cover

        task_name = inquirer.text(
            message="Name of your task?",
            validate=StringNonEmptyValidator(min_length=1, max_length=256),
        ).execute()
        task_type = inquirer.select(
            message='Type of your task?',
            choices=[{'name': titleize(name), 'value': name} for name in list_task_types()],
        ).execute()
        licence_ = hf_licence(
            message='Licence of your dataset?',
        ).execute()
        is_nsfw = inquirer.confirm(
            message='Is this task or samples not for all audience (NFAA) or not safe or work (NSFW)?',
            default=False,
        ).execute()

        params = init_cli(task_type)
        readme_metadata = copy.deepcopy(params.get('readme_metadata') or {})
        tags_to_add = []
        if is_nsfw:
            tags_to_add.append('not-for-all-audiences')
        if tags_to_add and 'tags' not in readme_metadata:
            readme_metadata['tags'] = []
        readme_metadata['tags'] = list(unique([*readme_metadata['tags'], *tags_to_add]))
        readme_metadata['license'] = licence_
        params['readme_metadata'] = readme_metadata

        init_params = dict(
            task_type=task_type,
            task_name=task_name,
            **params,
        )
        if 'force' in init_params:
            del init_params['force']

        try:
            fn_init(**init_params, force=False)
        except RepoAlreadyExistsError:
            confirm_clear = inquirer.confirm(
                message='This repository seems to exist.\n'
                        'if you insist on initializing it, the existing data will be cleaned.\n'
                        'Are you sure?'
            ).execute()
            if confirm_clear:
                fn_init(**init_params, force=True)

    return cli
