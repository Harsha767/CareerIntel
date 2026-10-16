.PHONY: setup test lint audit
setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
test:
	pytest -q
lint:
	ruff check src tests
audit:
	python -m careerintel.data.audit data/raw/$(FILE)
