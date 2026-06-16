# ---------------------------------------------------------------------------
# GSF image build / publish helpers
# ---------------------------------------------------------------------------
# These targets wrap the `docker buildx build` invocations needed to build
# the backend and frontend images and (optionally) push them to NVIDIA's NGC
# container registry. See `helm/gsf/README.md` for the full deploy story;
# this file only covers building and publishing the images.
#
# Common flow (one-time):
#
#   docker login nvcr.io                       # or: make login NGC_TOKEN=...
#   make build  REGISTRY=nvcr.io/<org>/<team>  # local builds (single arch)
#   make publish REGISTRY=nvcr.io/<org>/<team> # multi-arch push
#
# Variables (override on the command line):
#
#   REGISTRY        Full nvcr.io path, e.g. nvcr.io/nvidian/sw-gsf
#                   (required for `publish*` targets)
#   TAG             Image tag. Defaults to `git describe`.
#   NEMO            Path to a local NeMo-Retriever clone. Defaults to
#                   ../NeMo-Retriever. Pass NEMO=stub to build with the
#                   committed nemo-retriever stub instead (image starts and
#                   serves /api/health but raises NotImplementedError on
#                   real chat/data calls).
#   PYTHON_API_URL  Backend URL baked into the frontend's Next.js rewrites
#                   at build time. Defaults to http://gsf:3001
#                   (matches the in-cluster service the helm chart creates).
#   PLATFORMS       Comma-separated platforms for multi-arch publish.
#                   Defaults to linux/amd64,linux/arm64.
# ---------------------------------------------------------------------------

REGISTRY       ?=
TAG            ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
NEMO           ?= ../NeMo-Retriever
PYTHON_API_URL ?= http://gsf:3001
PLATFORMS      ?= linux/amd64,linux/arm64

BACKEND_IMAGE  := gsf
FRONTEND_IMAGE := frontend

# --build-context for NeMo-Retriever: pass `NEMO=stub` to skip and use the
# committed vendor/nemo_retriever_stub/ default baked into the Dockerfile.
ifeq ($(NEMO),stub)
  NEMO_CTX_ARG :=
else
  NEMO_CTX_ARG := --build-context nemo=$(NEMO)
endif

# --tag prefix. Plain image name when REGISTRY is empty (local builds);
# fully-qualified when REGISTRY is set (publish builds).
ifeq ($(REGISTRY),)
  BACKEND_REF  := $(BACKEND_IMAGE):$(TAG)
  FRONTEND_REF := $(FRONTEND_IMAGE):$(TAG)
else
  BACKEND_REF  := $(REGISTRY)/$(BACKEND_IMAGE):$(TAG)
  FRONTEND_REF := $(REGISTRY)/$(FRONTEND_IMAGE):$(TAG)
endif

.PHONY: help login builder build build build-frontend \
        publish publish publish-frontend

help:
	@echo "GSF image targets:"
	@echo "  make build               Build both images locally (host arch only)"
	@echo "  make build       Build backend image locally"
	@echo "  make build-frontend      Build frontend image locally"
	@echo "  make publish             Build + push both images (multi-arch)"
	@echo "  make publish     Build + push backend image"
	@echo "  make publish-frontend    Build + push frontend image"
	@echo "  make login NGC_TOKEN=... docker login nvcr.io via NGC API token"
	@echo
	@echo "Required for publish*: REGISTRY=nvcr.io/<org>/<team>"
	@echo "Current REGISTRY=$(REGISTRY) TAG=$(TAG) NEMO=$(NEMO)"

login:
	@if [ -z "$(NGC_TOKEN)" ]; then \
	  echo "Usage: make login NGC_TOKEN=<your NGC API token>"; \
	  echo "Get a token at https://ngc.nvidia.com/setup/api-key"; \
	  exit 2; \
	fi
	@echo "$(NGC_TOKEN)" | docker login nvcr.io --username '$$oauthtoken' --password-stdin

# Ensure a docker-container builder exists for multi-arch builds.
builder:
	@docker buildx inspect gsf-builder >/dev/null 2>&1 \
	  || docker buildx create --name gsf-builder --driver docker-container --use
	@docker buildx use gsf-builder

build:
	docker buildx build $(NEMO_CTX_ARG) -t $(BACKEND_REF) --load .

build-frontend:
	docker buildx build --build-arg PYTHON_API_URL=$(PYTHON_API_URL) \
	    -t $(FRONTEND_REF) --load ./frontend

build: build build-frontend

publish: require-registry builder
	docker buildx build --platform $(PLATFORMS) $(NEMO_CTX_ARG) \
	    -t $(BACKEND_REF) --push .

publish-frontend: require-registry builder
	docker buildx build --platform $(PLATFORMS) \
	    --build-arg PYTHON_API_URL=$(PYTHON_API_URL) \
	    -t $(FRONTEND_REF) --push ./frontend

publish: publish publish-frontend

# Internal: fail fast if REGISTRY is missing on a publish target.
.PHONY: require-registry
require-registry:
	@if [ -z "$(REGISTRY)" ]; then \
	  echo "ERROR: REGISTRY is required for publish targets."; \
	  echo "       e.g. make publish REGISTRY=nvcr.io/nvidian/sw-gsf"; \
	  exit 2; \
	fi
