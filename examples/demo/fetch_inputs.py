"""Downloading of input files and folders containing input data obtained from
the HEET reservoir and catchment delineation tool for the Myanmar case study 
from Google GDrive"""
import argparse
import os
import zipfile
from reemission.download import download_from_url, GDriveFileDownloader


def download_mya_case_study_inputs(
        url_link: str, 
        zipped_file: str,
        verbose: bool = True) -> None:
    """Download input data from external link provided in argument url_link.
    Assumes the url_link contains a zipped file that needs to be extracted
    into the folder where the zip file has been downloaded into."""
    download_from_url(
        url=url_link,
        output_path=zipped_file,
        downloader=GDriveFileDownloader(),
        update=True,
        relative_path=False,
        checksum=None,
        verbose=verbose)
    
    with zipfile.ZipFile(zipped_file, "r") as zip_ref:
        directory_path = os.path.dirname(zipped_file)
        zip_ref.extractall(directory_path)
    os.remove(zipped_file)


def main() -> None:
    """Simple command line argument parsing interfacing for calling file/folder
    content downloader for Myanmar case study delineations from command line
    using additional arguments."""
    parser = argparse.ArgumentParser(description="Download reemission demo inputs")
    parser.add_argument("url", type=str, help="URL link for downloading the inputs")
    parser.add_argument("-o", "--output", type=str, help="Output file path for the downloaded zipped file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    url_link = args.url
    output_file_path=args.output
    verbose = args.verbose
    download_mya_case_study_inputs(url_link, output_file_path, verbose)


if __name__ == "__main__":
    main()
