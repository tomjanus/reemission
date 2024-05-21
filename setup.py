""" Runner of the setup method with configuration given in setup.cfg """
import io
import re
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext
from setuptools import find_packages
from setuptools import setup


def read(*names, **kwargs):
    """ Function for reading text files """
    with io.open(join(dirname(__file__), *names),
                 encoding=kwargs.get('encoding', 'utf8')) as file_handle:
        return file_handle.read()

if __name__ == '__main__':
    # The use_scm_version option indicates that we want to use the
    # setuptools_scm package to set the version automatically based on git
    # tags, which will produce version strings such as 0.13 for a stable
    # release, or 0.16.0.dev113+g3d1a8747 for a developer version.
    
    setup(
        name="reemission",
        version='1.0.0',
        description='Python package for calculating greenhouse gas emissions from reservoirs',
        long_description='{}\n{}'.format(
            re.compile('^.. start-badges.*^.. end-badges', re.M | re.S).sub('', read('README.rst')),
            re.sub(':[a-z]+:`~?(.*?)`', r'``\1``', read('CHANGELOG.rst')),
            ),
        author='Tomasz Janus (dev), Aung Kyaw Kyaw (dev), Chris Barry (methodology)',
        author_email='tomasz.k.janus@gmail.com',
        url='https://github.com/tomjanus/reemission',
        #packages=find_packages(),
        packages = [
            'reemission.tests', 
            'reemission.examples', 
            'reemission.cli',
            'reemission.data_models',
            'reemission.integration',
            'reemission.postprocessing',
            'reemission'],
        package_dir={
            "reemission":"src/reemission",
            "reemission.tests":"tests",
            "reemission.examples":"examples",
            "reemission.cli":"src/reemission/cli",
            "reemission.data_models":"src/reemission/data_models",
            "reemission.integration":"src/reemission/integration",
            "reemission.postprocessing":"src/reemission/postprocessing"
        },
        #py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
        include_package_data=True,
        exclude_package_data={
            'reemission.examples': [
                ".shx", ".shp", ".prj", ".dbf", "cpg", "pdf", ".qmd", ".geojson", 
                ".html", ".csv", ".tex", ".xlsx"
            ]
        },
        zip_safe=False
    )
# use_scm_version=True,
# setup_requires=['setuptools_scm']
