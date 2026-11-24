# Contributing to Neospike

Thanks for helping out! Bug reports, fixes and ideas are all welcome.

## Getting started

```bash
git clone https://github.com/Daniel-Heo/hermes-neospike.git
cd hermes-neospike
uv sync
uv run pytest
```

You don't need Hermes installed to run the tests.

## Pull requests

- For larger changes, please open an issue first so we can agree on the approach.
- Keep each PR focused on one thing, and add or update tests for what you changed.
- Make sure `uv run pytest` passes.
- Update the README if user-visible behaviour changes.

A few things to keep in mind:

- Neospike is local-first: please don't add telemetry or new network calls.
- Tests should use a temporary `HERMES_HOME` and must not run the real `hermes`, `claude`, `uv` or `hf` commands.

## Reporting bugs

Open an issue with your OS, Hermes and Neospike versions, the output of `python install.py --check`, and steps to reproduce. Please don't attach your `memory.db` or other personal data.

## License

By contributing, you agree that your contributions are licensed under [Apache-2.0](LICENSE).
