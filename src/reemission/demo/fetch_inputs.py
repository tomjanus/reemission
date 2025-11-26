#    This file is part of Re-Emission.
#
#    Re-Emission is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Re-Emission is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Re-Emission.  If not, see <http://www.gnu.org/licenses/>.

"""Downloading of input files and folders containing input data obtained from
the GeoCARET reservoir and catchment delineation tool for the Myanmar case study 
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
