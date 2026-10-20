# From zero: start MySQL → generate data → load MySQL → pipeline → tests → figures → dashboard.
#
#   make all CINELUME_SOURCE=/path/to/streaming
#
# CINELUME_SOURCE is the Cinelume folder (gerar_dados.py, schema.sql). SCALE
# picks the generator scale: small, medium, full, or x60/x90 (see
# docs/DATA_SOURCE.md); x60 is what reports/ was produced with.

PYTHON ?= .venv/bin/python
CINELUME_SOURCE ?= $(HOME)/Downloads/datasets-github/streaming
SCALE ?= x60
DATA ?= $(CURDIR)/data/cinelume
COMPOSE = CINELUME_SOURCE=$(CINELUME_SOURCE) CINELUME_DATA=$(DATA) docker compose -f infra/docker-compose.yml

export CINELUME_SOURCE
export CINELUME_DATA = $(DATA)

.PHONY: all venv mysql-up data mysql-load pipeline pipeline-mysql test figures dashboard validate clean-mysql

all: mysql-up data mysql-load pipeline test figures dashboard

venv:
	python3.11 -m venv .venv && .venv/bin/pip install -e ".[dev]"

mysql-up:
	$(COMPOSE) up -d --wait mysql

data:
	$(PYTHON) scripts/generate_data.py --source $(CINELUME_SOURCE) --scale $(SCALE) --saida $(DATA)

mysql-load:
	$(COMPOSE) --profile carga run --rm carga

pipeline:
	$(PYTHON) -m churn.pipeline --rebuild

pipeline-mysql:
	$(PYTHON) -m churn.pipeline --rebuild --backend mysql

test:
	$(PYTHON) -m pytest -q

figures:
	$(PYTHON) -m churn.figures

dashboard:
	$(PYTHON) -m churn.dashboard

# Post-hoc only: compares the finished model with the simulation's answer key.
validate:
	$(PYTHON) -m churn.validation

clean-mysql:
	$(COMPOSE) down -v
