environment:
	@poetry config virtualenvs.in-project 1
	@poetry config virtualenvs.path `conda info --base`
	@poetry config keyring.enabled 0
	@poetry install

link_hf:
	@ln -s /opt/huggingface/ ~/.cache/huggingface

clear_pycache:
	@find . | grep -E "(__pycache__|\.pyc|\.pyo$\)" | xargs rm -rf

diagnose:
	@pre-commit run --all-files
