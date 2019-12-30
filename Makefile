.PHONY: check venv

check:
	flake8 rhasspyasr_kaldi_hermes/*.py
	pylint rhasspyasr_kaldi_hermes/*.py
	mypy rhasspyasr_kaldi_hermes/*.py

venv:
	rm -rf .venv/
	python3 -m venv .venv
	.venv/bin/pip3 install wheel setuptools
	.venv/bin/pip3 install -r requirements_all.txt
