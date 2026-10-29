.PHONY: all fetch build report test lint format

all:
	uv run fremtpl2 all

fetch:
	uv run fremtpl2 fetch

build:
	uv run fremtpl2 build

report:
	uv run fremtpl2 report

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src

format:
	uv run ruff format .
	uv run ruff check . --fix
