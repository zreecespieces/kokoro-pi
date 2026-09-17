# Convenience targets. The installer does all of this for you; these are for hacking.
PREFIX ?= $(HOME)/.kokoro-pi
MODELS ?= $(PREFIX)/models
VENV   ?= $(PREFIX)/venv
PYTHON ?= $(VENV)/bin/python
PORT   ?= 8080
THREADS ?= 4
export PYTHONPATH = $(CURDIR)/src

.PHONY: help venv native build serve validate samples clean distclean

help:
	@echo "make venv      create the virtual environment and install dependencies"
	@echo "make native    compile the operator library into $(MODELS)"
	@echo "make build     download the upstream model and derive the optimised ones"
	@echo "make serve     run the service in the foreground on port $(PORT)"
	@echo "make validate  measure quality and speed across variants"
	@echo "make samples   write a WAV per held-out utterance per variant into samples/"
	@echo "make clean     remove derived models, keeping the upstream download"

venv:
	$(PYTHON) -c '' 2>/dev/null || python3 -m venv $(VENV)
	$(VENV)/bin/pip install --quiet --upgrade pip wheel
	$(VENV)/bin/pip install --quiet -r requirements.txt

native:
	mkdir -p $(MODELS)
	bash native/build.sh $(MODELS)/libkokoro_pi_ops.so

build: venv
	$(PYTHON) -m kokoro_pi build --models $(MODELS) --threads $(THREADS)

serve:
	$(PYTHON) -m kokoro_pi serve --models $(MODELS) --port $(PORT) --threads $(THREADS)

validate:
	$(PYTHON) -m kokoro_pi validate --models $(MODELS) --variants upstream float int8 \
	  --reference upstream --threads $(THREADS)

samples:
	$(PYTHON) -m kokoro_pi validate --models $(MODELS) --variants upstream float int8 \
	  --reference upstream --audio-dir samples --repeats 1 --threads $(THREADS)

clean:
	rm -f $(MODELS)/kokoro-fused.onnx $(MODELS)/kokoro-int8.onnx \
	      $(MODELS)/kokoro-calibration.onnx $(MODELS)/calibration.npz $(MODELS)/models.json

distclean:
	rm -rf $(PREFIX)
