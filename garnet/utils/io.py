# coding: utf-8

"""
@File   : io.py
@Author : garnet
@Time   : 2020/10/18 10:49
"""

import json
import typing
import pathlib


def check_suffix(file_path, suffix):
    r"""Whether the file name ends with specified suffix.
    """
    suffix = suffix if suffix.startswith(".") else "." + suffix
    path = pathlib.Path(file_path)
    return path.suffix == suffix


def check_assert_suffix(file_path, suffix=None):
    if not suffix and not check_suffix(file_path, suffix):
        assert "A {} file must be offered, got {}".format(suffix, file_path)


def check_create_parents(file_path: typing.Union[str, pathlib.Path]):
    path = pathlib.Path(file_path) if isinstance(file_path, str) else file_path
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def safe_save(file_path, text, mode='w', encoding='utf-8', suffix=None):
    r"""Save string or binary bytes to file. Parent directory will be created if not exist.

    :param file_path: File path to save
    :param text: Content
    :param mode: Use `wb` if text is binary bytes, while `w` if text is string
    :param encoding: Encoding
    :param suffix: If `suffix` is not `None`, suffix check will be performed on `file_path`
    """
    path = pathlib.Path(file_path)
    check_assert_suffix(path, suffix)
    check_create_parents(path)

    with open(path, mode, encoding=encoding) as f:
        f.write(text)


def safe_save_json(file_path, dict_data):
    r"""Save `dict` object or list of `dict` objects into `.json` file.
    """
    path = pathlib.Path(file_path)
    check_assert_suffix(path, 'json')
    check_create_parents(path)

    with open(path, 'w', encoding='utf-8') as f:
        if isinstance(dict_data, dict):
            json.dump(dict_data, f, indent=4, ensure_ascii=False)
        else:
            f.write('\n'.join([json.dumps(d).replace('\n', '').replace('\r', '') for d in dict_data]))
