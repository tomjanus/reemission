# Makefile for reemission project
# Generate documentation
doc:
	rm -r docs/_build
	cd docs && make html

# Run tests
test:
	python -m unittest discover tests

# Code linting with pylint and flake8
lint:
	pylint src
	flake8 src

# Create a distribution wheel
wheel:
	rm -rf build
	rm -rf dist
	python setup.py sdist bdist_wheel

# Install with requirements
init:
	pip install -r requirements.txt

# Install in development mode
dev:
	pip install -e .

# Test upload to PyPi
upload_test:
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

# Upload distribution to PyPi
upload:
	twine upload dist/*
