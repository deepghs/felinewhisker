import os
import shutil
import tempfile
from unittest.mock import patch

import pytest

from felinewhisker.utils import clear_directory


# Create a temporary directory fixture
@pytest.fixture
def temp_dir():
    dir = tempfile.mkdtemp()
    yield dir
    shutil.rmtree(dir)


# Test class for clear_directory function
@pytest.mark.unittest
class TestClearDirectory:
    def test_clear_empty_directory(self, temp_dir):
        assert os.listdir(temp_dir) == []  # Ensure directory is empty
        clear_directory(temp_dir)
        assert os.listdir(temp_dir) == []  # Directory should still be empty

    def test_clear_directory_with_files(self, temp_dir):
        # Create files in the directory
        file_names = ["file1.txt", "file2.txt"]
        for file_name in file_names:
            with open(os.path.join(temp_dir, file_name), "w") as f:
                f.write("Dummy content")

        assert sorted(os.listdir(temp_dir)) == sorted(file_names)  # Ensure files are created
        clear_directory(temp_dir)
        assert os.listdir(temp_dir) == []  # Directory should be empty

    def test_clear_directory_with_subdirectories(self, temp_dir):
        # Create subdirectories
        sub_dir_names = ["subdir1", "subdir2"]
        for sub_dir_name in sub_dir_names:
            os.mkdir(os.path.join(temp_dir, sub_dir_name))

        assert sorted(os.listdir(temp_dir)) == sorted(sub_dir_names)  # Ensure subdirectories are created
        clear_directory(temp_dir)
        assert os.listdir(temp_dir) == []  # Directory should be empty

    def test_clear_directory_with_mixed_content(self, temp_dir):
        # Create a mix of files and directories
        os.mkdir(os.path.join(temp_dir, "subdir"))
        with open(os.path.join(temp_dir, "file.txt"), "w") as f:
            f.write("Dummy content")

        assert sorted(os.listdir(temp_dir)) == ["file.txt", "subdir"]  # Ensure contents are created
        clear_directory(temp_dir)
        assert os.listdir(temp_dir) == []  # Directory should be empty

    def test_clear_directory_unlink_failure(self, temp_dir):
        # Create a file and patch os.unlink to raise an exception
        file_path = os.path.join(temp_dir, "file.txt")
        with open(file_path, "w") as f:
            f.write("Dummy content")

        with patch('os.unlink', side_effect=Exception("Mocked exception")):
            with pytest.raises(Exception) as exc_info:
                clear_directory(temp_dir)
            assert str(exc_info.value) == "Mocked exception"
