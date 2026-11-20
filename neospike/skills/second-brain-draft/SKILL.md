---
name: second-brain-draft
description: Use when drafting a write-up from local notes, cited.
---

# Second brain: draft from notes

Write a draft document from a bundle of local notes and memories, citing a [[source]] for
every sentence, and save it as an agent note tagged "draft".

Use this skill when the user asks you to write a draft, summary, or write-up on a topic
from what is already stored in local memory (notes and memories).

## Procedure

1. **Search.** Run `neospike_memory_search` with the user's topic or question. Look at memories and
   automatic note chunks (kind `note_chunk`) first: they hold the stored facts and decisions
   themselves. Run more searches with narrower terms if the first one
   is thin.
2. **Read the sources.** Open each chosen note with `neospike_note_read` (by id or title).
   Read the memories returned by the search. Skip a source you have not actually read.
3. **Write the draft.** Every sentence carries the source it is based on:
   - a note as `[[Note title]]`, a memory as `[[mem:N]]` (N is the memory id).
   - Do not write any claim that is not in a source. If the sources do not cover a point,
     leave it out or list it under an "Open questions" heading without a claim.
   - Do not present a superseded memory as the current decision. Use the memory that
     replaced it; mention the old one only as history.
4. **Save.** Call `neospike_note_write` with title `DRAFT: <topic>` and tags `["draft"]`.
   The note is saved as an agent note. A draft without at least one `[[...]]` link is
   rejected; add the source links and save again.
5. **Report.** Tell the user the note title and how many distinct sources the draft cites.
   The user reads and edits the draft in the Notes tab.
