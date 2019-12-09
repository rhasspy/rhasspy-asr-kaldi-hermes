.PHONY: check

check:
	flake8 rhasspyasr_kaldi_hermes/*.py
	pylint rhasspyasr_kaldi_hermes/*.py
