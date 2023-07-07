"""Fetching binary and large-volume data from Google Drive."""
from typing import Union, Protocol, Optional, Any
import sys
from dataclasses import dataclass
import pathlib
import logging
import gdown
from reemission.app_logger import create_logger
from reemission.utils import get_package_file, split_path, is_directory, md5

# Create a logger
logger = create_logger(logger_name=__name__)


def file_valid(
        file_path: Union[str, pathlib.Path], 
        valid_hash: str, 
        chunk_size: int = 4) -> bool:
    """
    Validates a file against an MD5 hash value
    Args:
        file_path: path to the file for hash validation
        hash: str MD5 sum to validate the file against
    """
    return md5(file_path, chunk_size) == valid_hash


def create_directory_tree(path: pathlib.Path, verbose: bool = True) -> None:
    """For a given path, checks if the directory structure is present. If it is
    not, create the directory path from the first directory (counting from top)
    that is absent"""
    if verbose:
        logger.info("Creating folder structure in %s", path.as_posix())
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        if verbose:
            logger.info("Folder structure already exists.")


class URL_Downloader(Protocol):
    """ """
    def __call__(self, url: str, output_path: Union[pathlib.Path, str]) -> None:
        """Download data from url to output destination (path) """


@dataclass
class GDriveDirDownloader:
    """URL downloader using gdown's `download_folder' functionality. 
    Automatically overwrites whatever is already available.
    Does not check checksums during download."""
    quiet: bool = False
    remaining_ok: bool = True

    def __call__(self, url: str, output_path: Union[pathlib.Path, str]) -> None:
        """Download directory from Google Drive using a URL (shared link)"""
        if isinstance(output_path, pathlib.Path):
            output_path = output_path.as_posix()
        if not is_directory(output_path):
            logger.info("Downloader expects the path that is a directory not a file.")
            return
        gdown.download_folder(
            url=url, output=output_path, quiet=self.quiet, 
            remaining_ok=self.remaining_ok)


@dataclass
class GDriveFileDownloader:
    """ """
    quiet: bool = False
    fuzzy: bool = True
    resume: bool = False

    def __call__(self, url: str, output_path: Union[pathlib.Path, str]) -> None:
        """Download file from Google Drive using a URL to output destination 
        (path) """
        if isinstance(output_path, pathlib.Path):
            output_path = output_path.as_posix()
        if is_directory(output_path):
            logger.info(
                "Downloader expects the path to a file, not to a directory.")
            return
        dir_tree, _ = split_path(output_path)
        create_directory_tree(dir_tree, verbose=False)
        gdown.download(
            url, output=output_path, quiet=self.quiet, fuzzy=self.fuzzy,
            resume=self.resume)


@dataclass
class GDriveCachedFileDownloader:
    """ """
    quiet: bool = False
    md5: Optional[Any] = None
    extract: bool = False

    def __call__(self, url: str, output_path: Union[pathlib.Path, str]) -> None:
        """Download cached file from Google Drive using a URL to output 
        destination (path) """
        if isinstance(output_path, pathlib.Path):
            output_path = output_path.as_posix()
        if self.extract:
            postprocess = gdown.extractall
        else:
            postprocess = None
        if is_directory(output_path):
            logger.info(
                "Downloader expects path pointing to a file not to a directory.")
            return
        gdown.cached_download(
            url=url, path=output_path, md5=self.md5, quiet=self.quiet,
            postprocess=postprocess)


def download_from_url(
        url: str, output_path: pathlib.Path, downloader: URL_Downloader, 
        update: bool = True, relative_path: bool = False,
        checksum: Optional[Any] = None,
        verbose: bool = False,
        post_checksum_check: bool = False) -> None:
    """Download data from a URL
    Args:
        url: url pointing to date, e.g. share link from Google Drive
        output_path: directory/file relative to package root directory
        update: updates old data when checksum does not match
        relative_path: if True the path provided will be relative to the package
            root folder
        verbose: print more detailed output
        checksum: If given validate the file against the md5 sum
        post_checksum_check: Check the checksum once again, if given
    """
    # Create a separate function logger
    local_logger = create_logger(logger_name="Download_URL")
    if verbose:
        local_logger.setLevel(logging.DEBUG)
    else:
        local_logger.setLevel(logging.INFO)
    if relative_path:
        output_path = get_package_file(output_path)
    else:
        output_path = pathlib.Path(output_path)
    is_destination_dir: bool = is_directory(output_path)
    destination_exists: bool = pathlib.Path.exists(output_path)
    local_logger.info("Downloading from url %s", url)
    if destination_exists and not is_destination_dir:
        if checksum is not None:
            if file_valid(output_path, checksum):
                local_logger.debug(
                    "Data from url %s already exists and file is valid", url)
                return
            if not update:
                local_logger.debug(
                    "Data from url %s alread exists and is outdated", url)
                local_logger.debug(
                    "To ovewrite, run the function with output flag set to true")
                return
            local_logger.debug(
                "Data from url %s already exists but file is outdated", url)
            local_logger.debug("Updating the file with the new version")
            downloader(url, output_path)    
        else:
            if not update:
                local_logger.debug(
                    "Data from url %s already exists", url)
                local_logger.debug(
                    "To overwrite, run the function with update flag set to true")
                return                  
            local_logger.debug(
                "Data from url %s alread exists, ovewriting", url)
            downloader(url, output_path)
    else:
        local_logger.info("Downloading from url %s", url)
        downloader(url, output_path)

    if post_checksum_check and checksum and not is_destination_dir:
        # Check the checksum of the downloaded file (does not work for folders)
        if file_valid(output_path, checksum):
            local_logger.info("File matches checksum, OK.")
        else:
            local_logger.warning("Downloaded file does not match the checksum.")


if __name__ == "__main__":
    """ """
