.PHONY: all fetch build report test lint format package score api docker

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

package:
	uv run fremtpl2 package

score:
	uv run fremtpl2 score --input examples/policies_example.csv --output examples/quotes_example.csv

api:
	uv run uvicorn fremtpl2_pricing.api.app:app --port 8000

docker:
	docker build -t fremtpl2-rating .
