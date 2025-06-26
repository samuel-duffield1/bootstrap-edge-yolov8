.PHONY: help install lint format type-check security-check test clean all-checks

help:  ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies including dev tools
	poetry install --with dev,test

lint:  ## Run all linting checks
	@echo "Running flake8..."
	poetry run flake8 src/ --max-line-length=120 --extend-ignore=E203,W503
	@echo "Running pylint..."
	poetry run pylint src/edge_yolo_demo/

format:  ## Format code with black and isort
	@echo "Running black..."
	poetry run black src/
	@echo "Running isort..."
	poetry run isort src/

format-check:  ## Check if code formatting is correct
	@echo "Checking black formatting..."
	poetry run black --check src/
	@echo "Checking isort formatting..."
	poetry run isort --check-only src/

type-check:  ## Run type checking with mypy
	@echo "Running mypy..."
	poetry run mypy src/

security-check:  ## Run security checks with bandit
	@echo "Running bandit..."
	poetry run bandit -r src/ -c pyproject.toml

test:  ## Run tests
	poetry run pytest

all-checks: format-check lint type-check security-check  ## Run all code quality checks

clean:  ## Clean up cache files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
