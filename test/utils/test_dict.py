import pytest

from felinewhisker.utils import dict_merge


@pytest.fixture
def dicts():
    return {
        'empty_dict': {},
        'dict_a': {'key1': 'value1', 'key2': 'value2'},
        'dict_b': {'key2': 'new_value2', 'key3': 'value3'},
        'nested_dict_a': {'key1': {'subkey1': 'subvalue1'}, 'key2': 'value2'},
        'nested_dict_b': {'key1': {'subkey1': 'new_subvalue1'}, 'key3': 'value3'}
    }


@pytest.mark.unittest
class TestDictMerge:
    def test_empty_dicts(self, dicts):
        result = dict_merge(dicts['empty_dict'], dicts['empty_dict'])
        assert result == {}

    def test_empty_with_non_empty(self, dicts):
        result = dict_merge(dicts['empty_dict'], dicts['dict_a'])
        assert result == dicts['dict_a']
        result = dict_merge(dicts['dict_a'], dicts['empty_dict'])
        assert result == dicts['dict_a']

    def test_non_empty_dicts(self, dicts):
        result = dict_merge(dicts['dict_a'], dicts['dict_b'])
        expected = {'key1': 'value1', 'key2': 'new_value2', 'key3': 'value3'}
        assert result == expected

    def test_nested_dicts(self, dicts):
        result = dict_merge(dicts['nested_dict_a'], dicts['nested_dict_b'])
        expected = {'key1': {'subkey1': 'new_subvalue1'}, 'key2': 'value2', 'key3': 'value3'}
        assert result == expected

    def test_non_dict_first_argument(self, dicts):
        assert dict_merge('not_a_dict', dicts['dict_a']) == dicts['dict_a']

    def test_non_dict_second_argument(self, dicts):
        assert dict_merge(dicts['dict_a'], 'not_a_dict') == 'not_a_dict'
