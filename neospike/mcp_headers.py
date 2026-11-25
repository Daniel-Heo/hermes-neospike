"""Header helper of the neospike-mcp HTTP server for Claude Code (headersHelper).

Run: <runtime python> -I <plugins>/neospike/mcp_headers.py <HERMES_HOME>
Prints {"Authorization": "Bearer <token>"} with the token NEOSPIKE_MCP_TOKEN of <HERMES_HOME>/.env, so the
token is never written into Claude Code's own settings. Exit 1 with a message on stderr when there is none.

Stdlib only (it runs as a script, with this folder first on sys.path), and importable as neospike.mcp_headers
so the server reads the token with the same parser.
"""
import json
import os
import sys

ENV_FILE_NAME = ".env"
TOKEN_VARIABLE = "NEOSPIKE_MCP_TOKEN"
AUTHORIZATION_HEADER = "Authorization"
BEARER_PREFIX = "Bearer "
_EXPORT_PREFIX = "export "
_QUOTES = "\"'"
EXIT_NO_TOKEN = 1


def env_file_path(home) -> str:
    return os.path.join(os.fspath(home), ENV_FILE_NAME)


def parse_env_value(line: str, name: str):
    """The value of `name=value` (optionally `export name=value`, quoted or not) on *line*; None otherwise."""
    text = line.strip()
    if text.startswith(_EXPORT_PREFIX):
        text = text[len(_EXPORT_PREFIX):].lstrip()
    key, separator, value = text.partition("=")
    if not separator or key.strip() != name:
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in _QUOTES:
        value = value[1:-1]
    return value


def read_env_token(env_path) -> str:
    """NEOSPIKE_MCP_TOKEN of the .env file at *env_path* (the last definition wins); "" when absent."""
    try:
        with open(env_path, encoding="utf-8-sig") as env_file:
            lines = env_file.read().splitlines()
    except (OSError, UnicodeDecodeError):
        return ""
    token = ""
    for line in lines:
        value = parse_env_value(line, TOKEN_VARIABLE)
        if value is not None:
            token = value
    return token


def authorization_headers(token: str) -> dict:
    return {AUTHORIZATION_HEADER: BEARER_PREFIX + token}


def main(argv) -> int:
    home = argv[0] if argv else os.environ.get("HERMES_HOME", "")
    token = read_env_token(env_file_path(home)) if home else ""
    if not token:
        sys.stderr.write(f"{TOKEN_VARIABLE} is not set in {env_file_path(home) if home else ENV_FILE_NAME}; "
                         "run install.py again\n")
        return EXIT_NO_TOKEN
    sys.stdout.write(json.dumps(authorization_headers(token)))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
