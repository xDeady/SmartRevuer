.PHONY: install run test lint clean

install:
	pip install -e ".[dev]"

run:
	python -m smartrevuer

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/

clean:
	rm -rf outputs/*.png build/ dist/ src/*.egg-info
