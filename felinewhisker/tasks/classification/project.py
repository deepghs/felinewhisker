import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from typing import List, Callable

import pandas as pd
import yaml
from PIL import Image
from hbutils.string import plural_word
from hfutils.utils import number_to_tag, hf_normpath

from ...utils import padding_align


def create_readme_for_classification(workdir: str, task_meta_info: dict, df_samples: pd.DataFrame,
                                     fn_load_image: Callable[[str], Image.Image]):
    samples_dir = os.path.join(workdir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)

    readme_metadata = task_meta_info['readme_metadata']
    readme_metadata['task_categories'] = ['image-classification']
    readme_metadata['size_categories'] = [number_to_tag(len(df_samples))]
    with open(os.path.join(workdir, 'README.md'), 'w') as f:
        print(f'---', file=f)
        yaml.dump(readme_metadata, f, default_flow_style=False, sort_keys=False)
        print(f'---', file=f)
        print(f'', file=f)

        print(f'# Image Classification - {task_meta_info["name"]}', file=f)
        print(f'', file=f)
        labels = task_meta_info['labels']
        print(f'{plural_word(len(labels), "label")}, {plural_word(len(df_samples), "sample")} in total, '
              f'listed as the following:', file=f)
        print(f'', file=f)

        sample_cnt = 8
        samples = []
        tp = ThreadPoolExecutor(max_workers=12)

        def _load_preview_image(sample_id, dst_img_file):
            image = padding_align(fn_load_image(sample_id), (512, 768), color='#00000000')
            os.makedirs(os.path.dirname(dst_img_file), exist_ok=True)
            image.save(dst_img_file)

        for label in labels:
            df_label = df_samples[df_samples['annotation'] == label]
            row = {
                'Label': label,
                'Samples': f'{len(df_label)} ({len(df_label) / len(df_samples) * 100.0:.1f}%)',
            }
            ids = df_label['id'].tolist()
            if len(ids) > sample_cnt:
                ids = random.sample(ids, k=sample_cnt)
            selected = df_label[df_label['id'].isin(ids)].to_dict('records')

            for i in range(sample_cnt):
                if i < len(selected):
                    selected_item = selected[i]
                    dst_image_file = os.path.join(samples_dir, label, f'{i}.webp')
                    tp.submit(_load_preview_image, selected_item['id'], dst_image_file)
                    row[f'Sample #{i}'] = f'![{label}-{i}]({hf_normpath(os.path.relpath(dst_image_file, workdir))})'
                else:
                    row[f'Sample #{i}'] = 'N/A'
            samples.append(row)

        tp.shutdown(wait=True)
        df_samples = pd.DataFrame(samples)
        print(df_samples.to_markdown(index=False), file=f)
        print(f'', file=f)


def init_project_for_classification(workdir: str, task_name: str, readme_metadata: dict, labels: List[str]):
    meta_file = os.path.join(workdir, 'meta.json')
    with open(meta_file, 'w') as f:
        json.dump({
            'name': task_name,
            'labels': labels,
            'readme_metadata': readme_metadata,
            'task': 'classification',
        }, f, indent=4, sort_keys=True, ensure_ascii=False)

    md_file = os.path.join(workdir, 'README.md')
    with open(md_file, 'w') as f:
        readme_metadata['task_categories'] = ['image-classification']
        readme_metadata['size_categories'] = [number_to_tag(0)]
        print(f'---', file=f)
        yaml.dump(readme_metadata, f, default_flow_style=False, sort_keys=False)
        print(f'---', file=f)
        print(f'', file=f)

        print(f'# Image Classification - {task_name}', file=f)
        print(f'', file=f)
        print(f'{plural_word(len(labels), "label")} in total, as the following:', file=f)
        print(f'', file=f)
        for label in labels:
            print(f'* `{label}`', file=f)
        print(f'', file=f)

        print(f'This repository is empty and work in progress currently.', file=f)
        print(f'', file=f)
