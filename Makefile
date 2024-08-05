.PHONY: docs test unittest resource

PYTHON := $(shell which python)

PROJ_DIR      := .
DOC_DIR       := ${PROJ_DIR}/docs
BUILD_DIR     := ${PROJ_DIR}/build
DIST_DIR      := ${PROJ_DIR}/dist
TEST_DIR      := ${PROJ_DIR}/test
TESTFILE_DIR  := ${TEST_DIR}/testfile
DATASET_DIR   := ${TESTFILE_DIR}/dataset
SRC_DIR       := ${PROJ_DIR}/felinewhisker
TEMPLATES_DIR := ${PROJ_DIR}/templates
RESOURCE_DIR  := ${PROJ_DIR}/resource

RANGE_DIR      ?= .
RANGE_TEST_DIR := ${TEST_DIR}/${RANGE_DIR}
RANGE_SRC_DIR  := ${SRC_DIR}/${RANGE_DIR}

COV_TYPES ?= xml term-missing

LESS_FILE = $(shell find ${SRC_DIR} -name '*.less')
CSS_FILE := $(patsubst %.less,%.css,$(LESS_FILE))

package:
	$(PYTHON) -m build --sdist --wheel --outdir ${DIST_DIR}
build:
	pyinstaller -D -F -n felinewhisker -c felinewhisker_cli.py
clean:
	rm -rf ${DIST_DIR} ${BUILD_DIR} *.egg-info
	rm -rf build dist felinewhisker.spec

test: unittest

unittest:
	UNITTEST=1 \
		pytest "${RANGE_TEST_DIR}" \
		-sv -m unittest \
		$(shell for type in ${COV_TYPES}; do echo "--cov-report=$$type"; done) \
		--cov="${RANGE_SRC_DIR}" \
		$(if ${MIN_COVERAGE},--cov-fail-under=${MIN_COVERAGE},) \
		$(if ${WORKERS},-n ${WORKERS},)

docs:
	$(MAKE) -C "${DOC_DIR}" build
pdocs:
	$(MAKE) -C "${DOC_DIR}" prod

build_css: ${CSS_FILE}
%.css: %.less
	lesscpy $< $@
build_clean:
	rm -rf ${CSS_FILE}

