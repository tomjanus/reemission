#!/usr/bin/env python
from setuptools import setup

if __name__ == '__main__':
    # The use_scm_version option indicates that we want to use the
    # setuptools_scm package to set the version automatically based on git
    # tags, which will produce version strings such as 0.13 for a stable
    # release, or 0.16.0.dev113+g3d1a8747 for a developer version.
    setup(use_scm_version=True)
