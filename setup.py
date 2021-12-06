from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='dam-emissions',
    version='0.0.1',
    long_description=long_description,
    long_description_content_type='text/markdown',
    description='Calculation tool for GHG gas emissions from hydroelectric reservoirs',
    packages = find_packages(),
    py_modules=['dam-emissions'],
    package_dir={"", "src"})
