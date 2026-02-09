.PHONY: all lint test

all: lint test

lint:
	uv run ruff check .
	PYTHONPATH=../llm-client uv run ty check .

test:
	uv run pytest -q
