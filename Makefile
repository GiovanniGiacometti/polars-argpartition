SHELL=/bin/bash

venv:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

install:
	unset CONDA_PREFIX && \
	source .venv/bin/activate && maturin develop

install-release:
	unset CONDA_PREFIX && \
	source .venv/bin/activate && maturin develop --release

pre-commit:
	cargo +nightly fmt --all && cargo clippy --all-features
	.venv/bin/python -m ruff check . --fix --exit-non-zero-on-fix
	.venv/bin/python -m ruff format polars_argpartition tests
	.venv/bin/mypy polars_argpartition tests

test:
	.venv/bin/python -m pytest tests

run: install
	source .venv/bin/activate && python run.py

run-release: install-release
	source .venv/bin/activate && python run.py

requirements:
	uv export --no-hashes --no-editable --format requirements-txt > requirements.txt

publish:
	# before running this command, make sure you have the correct version in Cargo.toml
	uv run maturin develop --release
	git tag -a v$(VERSION_TAG) -m "Release v$(VERSION_TAG)"
	git push origin v$(VERSION_TAG)

