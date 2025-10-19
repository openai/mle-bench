# Base image - use the mlebench environment
FROM mlebench-env

# Agent-specific environment name (already created in base, but we'll use it)
ARG CONDA_ENV_NAME=mleb
ARG AGENT_DIR=/home/agent
ENV AGENT_DIR=/home/agent

RUN mkdir -p /home/code /home/submission /home/logs

# Copy only requirements first (for caching)
COPY requirements.txt ${AGENT_DIR}/requirements.txt

# Install Python dependencies in the conda environment
RUN /opt/conda/bin/conda run -n ${CONDA_ENV_NAME} pip install --no-cache-dir -r ${AGENT_DIR}/requirements.txt && \
    /opt/conda/bin/conda clean -afy

# Clone MLAgentBench repo
RUN git clone https://github.com/JunShern/MLAgentBench.git ${AGENT_DIR}/MLAgentBench && \
    cd ${AGENT_DIR}/MLAgentBench && \
    git checkout 328c713c02ea43e63f2d11cd7fd1293acb58f09f

# Copy benchmark and additional files
COPY ./benchmark_dir ${AGENT_DIR}/MLAgentBench/MLAgentBench/benchmarks/mlebench
COPY additional_notes.txt /home/additional_notes.txt
COPY start.sh ${AGENT_DIR}/start.sh

# Set working directory
WORKDIR ${AGENT_DIR}/MLAgentBench