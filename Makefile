#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = s2t-tr-dev
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = uv run python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format





## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Download processed AMI dataset from Google Drive
.PHONY: download_ami
download_ami:
	uv run python -m src.data.get_processed -d ami

## Download processed VoxPopuli dataset from Google Drive
.PHONY: download_voxpopuli
download_voxpopuli:
	uv run python -m src.data.get_processed -d voxpopuli

## Run the main results pipeline for the AMI dataset
.PHONY: run_main_results_ami
run_main_results_ami:
	uv run python -m src.experiments.main_results experiments=main_results_ami

## Run the main results pipeline for the VoxPopuli dataset
.PHONY: run_main_results_voxpopuli
run_main_results_voxpopuli:
	uv run python -m src.experiments.main_results experiments=main_results_voxpopuli

## Run the main results pipeline for the VoxPopuli dataset V2
.PHONY: run_main_results_voxpopuli_v2
run_main_results_voxpopuli_v2:
	uv run python -m src.experiments.main_results experiments=main_results_voxpopuli_v2

## Run the main results pipeline for the VoxPopuli dataset V3
.PHONY: run_main_results_voxpopuli_v3
run_main_results_voxpopuli_v3:
	uv run python -m src.experiments.main_results experiments=main_results_voxpopuli_v3

## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) src/dataset.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
