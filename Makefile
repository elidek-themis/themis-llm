install:
	@uv venv
	@uv pip install -e .

install-dev:
	@uv venv
	@uv pip install -e .["dev","viz"]

link_hf:
	@ln -s /opt/huggingface/ ~/.cache/huggingface

clear_pycache:
	@find . | grep -E "(__pycache__|\.pyc|\.pyo$\)" | xargs rm -rf

diagnose:
	@pre-commit run --all-files

count:
	@find . -path './.venv' -prune -o -name '*.py' -print | xargs wc -l | sort -nr
