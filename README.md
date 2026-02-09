# Coding Agent

Provider-aligned coding agent loop built on top of `unified-llm`.

## Quick Start

```python
import asyncio
from unified_llm.client import Client
from coding_agent import Session, SessionConfig, LocalExecutionEnvironment, OpenAIProfile

async def main():
    client = Client.from_env()
    env = LocalExecutionEnvironment(working_dir="/path/to/repo")
    profile = OpenAIProfile(model="gpt-5.2-codex")
    session = Session(profile, env, client, SessionConfig())

    await session.submit("Read README.md")

asyncio.run(main())
```

## Notes

- Provider base prompts are placeholders; replace them with the exact prompts from the provider-aligned reference agents.
- Tool definitions match the specification and can be overridden via the tool registry.
