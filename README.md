# Neospike

![Neospike: knowledge that gathers, assembles, and grows on its own](assets/branding/neospike-hero.png)

**An AI agent whose knowledge gathers, assembles, and grows on its own. Second brain + RAG, for [Hermes Agent](https://github.com/NousResearch/hermes-agent).**

Neospike is a memory provider plugin for Hermes Agent. The agent saves short facts and decisions as it works, keeps long documents as notes, and indexes past conversations. Code then tidies that knowledge in the background without calling an LLM: related memories are linked, linked groups become topics, and a topic that grows large becomes a note. One daily stage is the exception: it asks your Hermes model whether similar preferences should be kept, merged or removed (it can be turned off). Embeddings are computed on your machine and everything is stored in one SQLite file.

> Status: used daily on Windows 10. Linux and macOS have not been verified on real machines.

## What you get

| Part | What it holds |
|---|---|
| **Memory** | One short fact, decision, preference or constraint per item (up to 2,000 characters), saved by the agent. |
| **Notes** | Long documents the agent writes: designs, reports and `TASK_NNN` work orders. |
| **Docs** | Notes you wrote yourself. The agent and background jobs can read them but not change them. |
| **Preferences** | Your standing preferences and constraints, kept apart from memories. |
| **Conversations** | Past Hermes chats, copied in and indexed for search. |
| **Topics** | Groups of related memories, made and named by code. |

On top of that storage:

- **Automatic recall.** Before each turn, up to 8 matching memories and note sections (cosine 0.61 or more) are added to the agent's context. Items the conversation already contains are skipped.
- **Preferences in context.** Preferences without a situation are shown once per conversation. Preferences tied to a situation are shown when the current message matches that situation, at most 3 per turn. A constraint always needs a situation, so it is never shown in every conversation.
- **Dashboard tab** in the Hermes web dashboard: Map (a 3D view of memories, notes and topics, with folders), Notes, Docs, Memory, Preferences, Conversations, Config and Setup. The interface is English or Korean, following the dashboard language.
- **MCP server** for Claude Code and other MCP clients: `mem_search`, `note_read`, `note_related`, `note_append_report`.
- **Markdown export** of notes as a zip. Export is one-way; the database stays the source of truth.
- **Backups.** The daily job writes a full snapshot first and incremental backups after that.

## How knowledge grows

1. **Saving.** The agent saves one fact per memory. Text over 2,000 characters, or with two or more headings, is stored as a note instead.
2. **At save time.** The new memory is compared with its 3 nearest neighbours. An older memory that says the same thing (cosine 0.90 or more, word counts within 2) is removed. Older items with cosine 0.80 or more are returned to the agent, which corrects or deletes them in one `neospike_memory_reconcile` call so that the newest content wins.
3. **Linking.** Each memory is linked to up to 3 others with cosine 0.6 or more. A memory with cosine 0.8 or more to a section of an automatic note joins that note.
4. **Topics.** A linked group of more than 5 memories without a topic becomes a topic. A topic with more than 10 memories becomes an automatic note with one section per memory, and those memories and the topic are removed.
5. **Your edits win.** A correction applies at once and the old text is kept as history. Automated jobs never change a note you wrote.

Steps 3 and 4 are plain code. They run after a save and in the daily job; no LLM is called. Preferences and constraints that are similar but not exact duplicates are judged once a day by an LLM, inside Hermes (see [Preference tidy-up](#preference-tidy-up)).

## Requirements

- [Hermes Agent](https://github.com/NousResearch/hermes-agent). The web dashboard is needed only for the Neospike tab.
- Python 3.11 or newer to run `install.py`. [uv](https://docs.astral.sh/uv/) on the `PATH` is needed only when an earlier version of `install.py` installed the package into the Hermes CLI environment (`hermes-agent/venv`); the installer uses it to remove that copy.
- About 600 MB of disk space for the embedding model.

The embedding model is BAAI/bge-m3 as an int8 ONNX file from `Xenova/bge-m3`, pinned to one revision. It produces 1,024-dimension vectors and reads up to 2,048 tokens per input. It is MIT-licensed and not gated, so no Hugging Face account is needed. The plugin does not ship the model; the installer downloads it.

On Windows the model runs on the graphics card through DirectML when that works, and on the CPU otherwise. macOS and Linux always use the CPU. No setting is needed: the embedding service tries the graphics card once when it starts and writes the result (`"device": "directml"` or `"cpu"`) to `$HERMES_HOME/neospike/.cache/inference.log`. On one test machine (RTX 3080) a 2,048-token text took about 0.4 seconds on the graphics card and about 2.9 seconds on the CPU. A short text takes about 0.3 seconds on the graphics card, which is slower than the CPU (under 0.1 seconds). Vectors from the two devices differ slightly (cosine similarity about 0.97 to 0.99); vectors already stored are kept and used as they are when the device changes.

## Install

```bash
git clone https://github.com/Daniel-Heo/hermes-neospike.git
cd hermes-neospike
python install.py          # python3 on Linux/macOS
```

Then restart the gateway and check the result:

```bash
hermes gateway restart
python install.py --check
```

Opening a new Desktop window is not enough: the running gateway keeps the old code loaded.

`install.py` does the following, in this order:

1. finds the Hermes runtime Python and, if an earlier version installed the `neospike` package into the Hermes CLI environment (`hermes-agent/venv`), removes that package (only that one) with `uv pip uninstall`;
2. copies the plugin to `$HERMES_HOME/plugins/neospike` (the new copy is staged in `$HERMES_HOME/neospike/.install/staging` and the existing copy is kept as `$HERMES_HOME/neospike/.install/previous`, outside `plugins`, because Hermes loads every folder there as a plugin; `neospike.previous` and `neospike.installing` folders an earlier version left in `plugins` are deleted) and adds it to `plugins.enabled`;
3. creates `MEMORY_RULE.md` for each Hermes home that uses Neospike, unless the file already exists;
4. enables the plugin so that the Hermes package manager installs `numpy`, `faiss-cpu`, `tokenizers`, `onnxruntime` (on Windows `onnxruntime-directml`, the same library with graphics card support) and `huggingface-hub` into the Hermes runtime;
5. downloads the model;
6. sets `memory.provider` to `neospike`;
7. registers the daily cron job `neospike-daily` (every day at 12pm);
8. registers the `neospike` MCP server in Claude Code at user scope.

| Option | Effect |
|---|---|
| `--home <path>` | Install into this Hermes home instead of the default one. |
| `--skip-models` | Do not download the model. |
| `--check` | Verify an existing installation and change nothing. |
| `--dry-run` | Print the planned copies and commands and change nothing. |
| `--uninstall` | Remove Neospike from the Hermes home (see [Uninstall](#uninstall)). With `--dry-run`, print the plan only. |

`$HERMES_HOME` is `%LOCALAPPDATA%\hermes` on Windows and `~/.hermes` on Linux and macOS. Data lives in `$HERMES_HOME/neospike`.

### Uninstall

To pause Neospike without removing it, turn it off in the dashboard Setup tab (`enabled` below). To remove it:

```bash
python install.py --uninstall --dry-run   # print the plan
python install.py --uninstall             # add --home <path> for a profile home
hermes gateway restart
```

`--uninstall` removes the cron job `neospike-daily` and the cron scripts in `$HERMES_HOME/scripts` (root home only), the Claude Code MCP server `neospike` (only when it points at that home), asks the embedding service to stop (unless another home still uses Neospike), and runs `hermes plugins remove neospike`, which deletes `$HERMES_HOME/plugins/neospike` and clears `plugins.enabled` and a `memory.provider` of `neospike`; `$HERMES_HOME/neospike/.install` (the previous plugin copy) and any `neospike.previous` or `neospike.installing` left in `plugins` are deleted too. Steps that find nothing to remove are skipped, so it can be run again. Your data (`$HERMES_HOME/neospike`) and the model (`$HERMES_HOME/models/bge-m3-onnx`) stay; the command prints both paths, and you delete those folders yourself if you no longer want them.

### Hermes profiles

Each profile keeps its own database. All profiles share the embedding service and the model of the root Hermes home, and the root home's daily job covers every profile. Installing into a profile (`--home <root>/profiles/<name>`) therefore registers no cron job and leaves the MCP entry pointing at the root home.

## Tools

The agent gets these tools from the provider:

| Tool | Purpose |
|---|---|
| `neospike_memory_save` | Save one memory, preference or constraint. It can also fix, in the same call, a recalled memory or automatic-note line that turned out wrong. |
| `neospike_memory_update` | Correct an item in place, or supersede it when a decision changed. |
| `neospike_memory_delete` | Remove one item. |
| `neospike_memory_reconcile` | Settle the older, similar items returned by a save. |
| `neospike_memory_search` | Search stored knowledge. |
| `neospike_note_read` | Read a whole note (agent or automatic). |
| `neospike_note_write` | Create or update a note. |
| `neospike_doc_read` | Read a whole doc (a note the user wrote). |
| `neospike_doc_write` | Edit an existing doc, only when the user asks. |

MCP clients get a smaller, mostly read-only set:

| Tool | Purpose |
|---|---|
| `mem_search` | Search memories, notes and docs at once. |
| `note_read` | Read one note or doc by id or exact title. |
| `note_related` | Backlinks, outgoing links and the most similar notes. |
| `note_append_report` | Append a work report to a `TASK_` note. |

## Configuration

Settings are in `$HERMES_HOME/neospike/config.json`.

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `true` | On/off switch (Setup tab). `false` pauses Neospike in this home without uninstalling it: no automatic recall, no conversation capture, no save follow-up, tool calls and MCP searches return an error, the daily job skips the home, and the embedding service is asked to stop unless another home uses it. Turning it back on works from the next turn, without a restart. |
| `conversation_capture` | `true` | Copy user and assistant turns into the conversation store, with secrets masked. |
| `conversation_prefetch` | `false` | Also add conversation excerpts to automatic recall. |
| `recall_log_enabled` | `true` | Write one log line per recall under `neospike/logs` (ids and scores only, never the query text). |
| `recall_log_months` | `12` | Months of recall logs to keep. |
| `turn_trace_enabled` | `false` | Add short Neospike diagnostics to chat replies. |
| `map_recent_days` | `7` | Age below which the Map shows an item as recent. |
| `map_stale_days` | `14` | Age above which the Map shows an item as old. |
| `extraction_language` | `en` | Language (`en` or `ko`) Neospike uses when it extracts memories from conversations. Set it in the Config tab. |
| `pref_reconcile_enabled` | `true` | Run the daily preference tidy-up (the Hermes idle task `neospike-pref-reconcile`), the only Neospike job that calls an LLM. |

The rules that tell the agent what to save are the text of `$HERMES_HOME/neospike/MEMORY_RULE.md`. Edit it in the Config tab or in any editor.

The model of the preference tidy-up is picked in the Config tab ("Preference and constraint cleanup model"). It is not in `config.json`: it is the Hermes auxiliary slot `auxiliary.neospike_pref_reconcile` (provider, model) of `$HERMES_HOME/config.yaml`, the same value the Hermes auxiliary model screen shows. `auto`, which `install.py` sets only when the slot is missing, follows the chat model. Models reached with an API key are marked ` API`.

Environment variables:

| Variable | Meaning |
|---|---|
| `HERMES_HOME` | The Hermes home to use. |
| `HERMES_PYTHON` | The Hermes Python to use when the installer cannot find it. |
| `HERMES_BGE_M3_DIR` | Folder that holds the model files, if not the default `models/bge-m3-onnx`. |
| `NEOSPIKE_INFERENCE_HOME` | Home whose embedding service all profiles share, if not the root home. |

## Daily job

`neospike-daily` runs every day at 12pm and runs these stages in order. All of them are code only; none calls an LLM.

1. `backup`: back up the database.
2. `link-prune`: remove links whose memory or note is gone.
3. `conv-import`: bring in conversations added or changed since the last run.
4. `conv-index`: index the new conversation text.
5. `embed-pending`: embed memories that were saved while the embedder was unavailable, then link them.
6. `topic-discover`: turn linked groups into topics.
7. `topic-to-note`: turn large topics into notes.

A failure is reported to Bot Chat. A successful run reports nothing.

The schedule is Hermes cron's. If Hermes is not running at 12pm, Hermes runs the missed job once when it runs again (`cron.catch_up_missed`, on by default). A run that started but was cut off, for example by shutting the computer down, is not repeated; the next run is the next day. `hermes cron run <job id>` runs it at once.

### Preference tidy-up

Hermes runs this as the plugin idle task `neospike-pref-reconcile`: at most once every 24 hours, and only after Hermes has been idle for 2 hours (the Hermes default). It finds preference or constraint pairs of the same user with cosine 0.8 or more that are not exact duplicates, and asks an LLM (32 pairs per call, at most 4 calls a run) to keep both, merge them, or delete one. A preference the user stated is never deleted because of one the agent inferred. Pairs judged "keep" are not asked again until one of them changes. A merged row is embedded by the next daily job.

- **Model.** It uses the LLM model selected in the Config tab ("Preference and constraint cleanup model"). The default follows your chat model.
- **When it is scheduled.** Only when Hermes loads Neospike as a general plugin as well as the memory provider: `plugin.yaml` declares `kind: standalone` and `install.py` adds `neospike` to `plugins.enabled`. If `neospike` is not in `plugins.enabled`, the tidy-up does not run.
- **Turning it off.** Set `pref_reconcile_enabled` to `false`, or pause Neospike (`enabled: false`). `hermes plugins idle-tasks neospike` shows the task; add `--pause neospike-pref-reconcile` to pause it. `neospike pref-reconcile` on the command line only explains this and exits 1.

## Command line

```bash
<runtime python> -I $HERMES_HOME/plugins/neospike/hermes_launch.py <hermes checkout> neospike.cli --home <home> <command>
```

`<runtime python>` is the first element that `hermes --print-runtime-command` prints (`install.py` saves it as `runtime_python` in `$HERMES_HOME/neospike/runtime.json`), and `<hermes checkout>` is the `hermes-agent` folder. `hermes_launch.py` starts Python the way the Hermes launcher does, so the command runs with the packages the Hermes package manager installed, and puts the plugin folder on the import path. The dashboard, the cron job, the MCP server and the embedding service run Neospike the same way. `--help` lists every command. The main groups are:

- memories: `memory-search`, `memory-count`, `memory-update`, `memory-delete`, `memory-revisions`, `memory-restore`
- notes: `note-save`, `note-read`, `note-search`, `note-delete`, `note-restore`
- preferences: `pref-add`, `pref-update`, `pref-delete`, `pref-list`
- conversations: `conv-import`, `conv-index`, `conv-search`, `conv-read`
- backup and maintenance: `backup`, `backup-restore`, `reembed`

A command prints its result as JSON. A failed command prints `{"error": ..., "message": ...}` and exits 1, also when the embedding model or the embedding service is not available (the message names the missing model file); a wrong argument prints the usage and exits 2.

A database embedded with a different model or dimension is refused until `reembed` rewrites its vectors. Back up `neospike/memory.db` first.

## Privacy

- Embedding, search and every background job run on your machine. Only the daily batch uses an LLM: it tidies preference data with the LLM model you selected, the same way your chat uses its model. Set `pref_reconcile_enabled` to `false` to stop it.
- Recalled text goes only to the model your agent already uses.
- Text that looks like a credential or a prompt-injection attempt is rejected when it is saved.
- Recalled text is given to the agent as data, not as instructions.

## Disclosure

What Neospike does outside its own data folder (`$HERMES_HOME/neospike`):

- **Network.** Two things contact a remote host. (1) The model download: `install.py` and the dashboard Setup tab run the Hugging Face CLI (`hf download Xenova/bge-m3` at a pinned revision). (2) The daily batch: preference data is extracted and reworked by the selected LLM model. The packages are installed by the Hermes package manager (`hermes plugins enable`), which downloads them from the package index. The embedding service listens only on a local socket (a Unix socket, or `127.0.0.1` with a per-request token on Windows). There is no telemetry, analytics or update check.
- **Reads outside the data folder.** `conv-import`, the provider and the dashboard open Hermes's `$HERMES_HOME/state.db` read-only to read past conversations. `install.py` reads `$HERMES_HOME/config.yaml` and `$HERMES_HOME/cron/jobs.json`. The daily cron script reads `config.yaml` of the root home and of each profile (`memory.provider`) to find the homes that use Neospike.
- **Shell commands and child processes.** All of them run without a shell. `install.py` runs `hermes --print-runtime-command` (to find the Hermes runtime Python), `uv pip uninstall neospike` (only when an earlier install left that package in the Hermes CLI environment), `hf download`, the environment, import and provider checks of `neospike.runtime_check`, `hermes plugins enable`, `hermes config set memory.provider neospike`, `hermes cron create` / `edit` / `remove`, and `claude mcp remove` / `add-json`; with `--check` it also runs `hermes memory status`; with `--uninstall` it runs `hermes cron remove`, `claude mcp get` / `remove`, `hermes plugins remove neospike` (or `hermes memory off` when the plugin folder is already gone). The dashboard runs `hermes config get` / `set memory.provider`, `hf download`, the Neospike CLI, and the dependency and provider checks of `neospike.runtime_check`. The provider checks the embedding service's dependencies with `neospike.runtime_check` before it reports itself available. The MCP server runs the Neospike CLI for writes. Every Python child (`hf download`, the CLI, the checks, the MCP server, the cron script's CLI call and the embedding service) runs on the Hermes runtime Python through `hermes_launch.py`, which imports Hermes's `hermes_bootstrap` to select the package manager's environment.
- **Long-running background process.** The embedding service (`neospike.inference`, started through `hermes_launch.py`) is started on first use as a detached process and keeps running after the agent exits; one service serves every profile of a Hermes home. It stops on Ctrl-C or SIGTERM, or when asked over its socket (with its token) by `install.py --uninstall` or by turning the Setup tab switch off, unless another home still uses it. Its socket and log (`inference.log`) are in `$HERMES_HOME/neospike/.cache`. Inside the provider, the conversation indexer and the save follow-up run as daemon threads.
- **Files written outside the data folder.** `install.py` copies the plugin to `$HERMES_HOME/plugins/neospike` (staged in, and the previous copy kept in, `$HERMES_HOME/neospike/.install`; a `neospike.previous` or `neospike.installing` folder an earlier install left in `$HERMES_HOME/plugins` is removed, because Hermes would load it as a second copy of the plugin), the cron scripts to `$HERMES_HOME/scripts`, the model to `$HERMES_HOME/models/bge-m3-onnx`, adds `neospike` to `plugins.enabled` in `$HERMES_HOME/config.yaml` (and, through `hermes config set`, `auxiliary.neospike_pref_reconcile.provider: auto` in each home that has no such slot), and registers the `neospike` MCP server in Claude Code at user scope through `claude mcp add-json`. `MEMORY_RULE.md` is created inside the data folder.
- **Credentials.** Neospike stores no tokens, API keys or passwords. The model is not gated, so no Hugging Face token is needed. The LLM call goes through Hermes, which holds the provider credentials; Neospike never reads them. The dashboard passes its child processes only a fixed list of environment variables and discards their output.

## Development

```bash
uv sync
uv run pytest
```

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). The model keeps its own licence.

## References

- [Hermes Agent](https://github.com/NousResearch/hermes-agent): the `MemoryProvider` and dashboard plugin interfaces.
- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) and the ONNX conversion [Xenova/bge-m3](https://huggingface.co/Xenova/bge-m3).
