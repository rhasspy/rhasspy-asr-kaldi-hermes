SHELL := bash
PYTHON_NAME = rhasspyasr_kaldi_hermes
PACKAGE_NAME = rhasspy-asr-kaldi-hermes
SOURCE = $(PYTHON_NAME)
PYTHON_FILES = $(SOURCE)/*.py *.py
SHELL_FILES = bin/* debian/bin/*

.PHONY: reformat check dist venv pyinstaller debian docker deploy sdist

version := $(shell cat VERSION)
architecture := $(shell bash architecture.sh)

# -----------------------------------------------------------------------------
# Python
# -----------------------------------------------------------------------------

reformat:
	scripts/format-code.sh $(PYTHON_FILES)

check:
	scripts/check-code.sh $(PYTHON_FILES)

venv:
	scripts/create-venv.sh

test:
	echo "No tests yet"

dist: sdist debian

sdist:
	python3 setup.py sdist

# -----------------------------------------------------------------------------
# Docker
# -----------------------------------------------------------------------------

docker: pyinstaller
	docker build . -t "rhasspy/$(PACKAGE_NAME):$(version)" -t "rhasspy/$(PACKAGE_NAME):latest"

deploy:
	echo "$$DOCKER_PASSWORD" | docker login -u "$$DOCKER_USERNAME" --password-stdin
	docker push "rhasspy/$(PACKAGE_NAME):$(version)"

# -----------------------------------------------------------------------------
# Debian
# -----------------------------------------------------------------------------

pyinstaller:
	scripts/build-pyinstaller.sh "${architecture}" "${version}"

debian:
	scripts/build-debian.sh "${architecture}" "${version}"

# -----------------------------------------------------------------------------
# Downloads
# -----------------------------------------------------------------------------

downloads: rhasspy-libs

# Rhasspy development dependencies
rhasspy-libs: $(DOWNLOAD_DIR)/rhasspy-silence-0.1.2.tar.gz $(DOWNLOAD_DIR)/rhasspy-hermes-0.1.6.tar.gz

$(DOWNLOAD_DIR)/rhasspy-silence-0.1.2.tar.gz:
	mkdir -p "$(DOWNLOAD_DIR)"
	curl -sSfL -o $@ "https://github.com/rhasspy/rhasspy-silence/archive/master.tar.gz"

$(DOWNLOAD_DIR)/rhasspy-hermes-0.1.6.tar.gz:
	mkdir -p "$(DOWNLOAD_DIR)"
	curl -sSfL -o $@ "https://github.com/rhasspy/rhasspy-hermes/archive/master.tar.gz"
