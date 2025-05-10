.PHONY: run install clean format install-deps

# Install the package in development mode
# Uses uv to install with copying of dependencies
install:
	uv pip install --link-mode=copy -e '.[dev]'

# Install dependencies from requirements.txt file
# Uses uv package manager for faster installation
install-deps:
	uv pip install -r requirements.txt

# Run the main application 
# Formats code before running
run: format
	python app/main.py

# List available models from the model runner service
# Uses curl to fetch and jq to format the JSON response
list-models:
	curl -s  http://model-runner.docker.internal/models | jq

# Format and lint the code using ruff
# Sets line length to 122 and applies fixes automatically
format:
	ruff format --line-length 120 app/
	ruff check --line-length 120 --fix app/

# Clean up generated files and caches
# Removes Python cache files and other temporary directories
clean:
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type d -name "*.egg-info" -exec rm -r {} +
	find . -type d -name .pytest_cache -exec rm -r {} +
	find . -type d -name .mypy_cache -exec rm -r {} +
	find . -type d -name .ruff_cache -exec rm -r {} +