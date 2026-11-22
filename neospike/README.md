# Neospike

This folder is the Neospike plugin as Hermes loads it: a `MemoryProvider`, a dashboard tab, an MCP server and a command line. It is installed outside the Hermes checkout, in `$HERMES_HOME/plugins/neospike`.

For what Neospike does and how to install it, see the [project README](https://github.com/Daniel-Heo/hermes-neospike#readme).

## Storage

- One SQLite file: `$HERMES_HOME/neospike/memory.db`. `$HERMES_HOME` is `%LOCALAPPDATA%\hermes` on Windows and `~/.hermes` on Linux and macOS.
- Each Hermes profile has its own database. All profiles share one embedding service and one copy of the model, in the root Hermes home.
- Settings are in `$HERMES_HOME/neospike/config.json`. The rules that tell the agent what to save are in `$HERMES_HOME/neospike/MEMORY_RULE.md`.

## Model

BAAI/bge-m3 as an int8 ONNX file from `Xenova/bge-m3`, pinned to one revision. It produces 1,024-dimension vectors, and the dimension is fixed in code.

An input may be up to 2,048 tokens. A longer document is an error and is never cut silently. A longer search query is cut to the first 2,048 tokens.

A database embedded with a different model or dimension is refused until this command rewrites its vectors. Back up the database first.

```sh
python -m neospike.cli --home <home> reembed
```

The plugin does not ship the model. bge-m3 is MIT-licensed and not gated.

## What is stored

- The agent stores a memory by calling `neospike_memory_save`. Entries written to the built-in Hermes memory files are mirrored into the database as well. Chat text is never turned into a memory automatically.
- User and assistant turns are copied into the conversation store, with secrets masked, when `conversation_capture` is on (the default). They are searchable but stay out of automatic recall unless `conversation_prefetch` is on.
- Preferences and constraints are kept in their own table. A preference without a situation is shown in every conversation, and only when the user said it directly. A constraint always needs a situation.
- Memories have no retention period. They stay until they are deleted, superseded or archived.
- Text that looks like a credential or a prompt-injection attempt is rejected before it is saved.

## Recall

Before each turn, up to 8 matching memories and note sections with cosine 0.7 or more are added to the agent's context. Items the conversation already contains are skipped. The block is labelled as untrusted data, not instructions.

## Agent tools

`neospike_memory_save`, `neospike_memory_update`, `neospike_memory_delete`, `neospike_memory_reconcile`, `neospike_memory_search`, `neospike_note_read`, `neospike_note_write`, `neospike_doc_read`, `neospike_doc_write`.

## MCP tools

`note_read`, `mem_search`, `mem_period`, `chat_search`, `chat_read`, `chat_period` (all read-only). The server needs the optional `mcp` dependency (`pip install 'neospike[mcp]'`).

## Command line

```sh
python -m neospike.cli --home <home> memory-count
python -m neospike.cli --home <home> memory-search "query"
python -m neospike.cli --help
```

## Backup and restore

The daily job runs `backup`. The first run writes a full snapshot (`full-<UTC>.db`), and later runs write only the rows that changed (`inc-<UTC>.zip`). A schema change starts a new chain with a full snapshot. Restore with `backup-restore`.
