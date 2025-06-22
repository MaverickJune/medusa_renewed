###############################################################################
# ---------- Stage 0 : build Conda env + compile Medusa -----------------------
###############################################################################
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ARG MICROMAMBA_VERSION=2.3.0
ENV  DEBIAN_FRONTEND=noninteractive \
     MAMBA_ROOT_PREFIX=/opt/conda \
     PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH \
     TZ=UTC

# --- base packages + tool-chain ---------------------------------------------
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        build-essential clang tmux curl ca-certificates bzip2 git && \
    rm -rf /var/lib/apt/lists/*

ENV CC=/usr/bin/gcc \
    CXX=/usr/bin/g++

# --- micromamba --------------------------------------------------------------
RUN curl -L "https://micro.mamba.pm/api/micromamba/linux-64/${MICROMAMBA_VERSION}" \
    | tar -xvj -C /usr/local/bin/ --strip-components=1 bin/micromamba

# --- create env --------------------------------------------------------------
COPY environment.yml /tmp/environment.yml
RUN micromamba env create -y -f /tmp/environment.yml && \
    micromamba clean --all --yes

ENV ENV_NAME=medusa

# --- flash-attn --------------------------------------------------------------
RUN micromamba run -n medusa \
    pip install --no-cache-dir flash-attn==2.8.0.post2

# --- source code -------------------------------------------------------------
COPY . /workspace/medusa
WORKDIR /workspace/medusa
RUN micromamba run -n medusa pip install --no-deps -e .

###############################################################################
# ---------- Stage 1 : runtime (small, still CUDA 12.4) -----------------------
###############################################################################
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV  DEBIAN_FRONTEND=noninteractive \
     MAMBA_ROOT_PREFIX=/opt/conda \
     PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH \
     TZ=UTC \
     CC=/usr/bin/gcc \
     CXX=/usr/bin/g++

# --- bring Conda env, micromamba, CUDA libs, source --------------------------
COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /usr/local/bin/micromamba /usr/local/bin/micromamba
COPY --from=builder /usr/local/cuda /usr/local/cuda
COPY --from=builder /workspace/medusa /workspace/medusa

# --- install compiler tool-chain & helpers in runtime stage ------------------
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        build-essential clang tmux git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/medusa
CMD ["/bin/bash"]
