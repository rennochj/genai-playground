.PHONY: run install clean

install:
	uv pip install --link-mode=copy -e '.[dev]'

run:
	python main.py

list-models:
	curl -s  http://model-runner.docker.internal/models | jq

clean:
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type d -name "*.egg-info" -exec rm -r {} +
	find . -type d -name .pytest_cache -exec rm -r {} +
	find . -type d -name .mypy_cache -exec rm -r {} +
	find . -type d -name .ruff_cache -exec rm -r {} +