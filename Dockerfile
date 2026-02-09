FROM ghcr.io/astral-sh/uv:python3.12-bookworm

WORKDIR /workspace
COPY coding-agent /workspace/coding-agent
COPY llm-client /workspace/llm-client

WORKDIR /workspace/coding-agent
ENV UV_LINK_MODE=copy
RUN uv sync

CMD ["bash", "-lc", "uv run coding-agent --provider openai --stream --show-tools --approve-tools"]
