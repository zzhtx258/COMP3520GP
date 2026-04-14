---
name: rag
description: Query HKU degree-plan and employment data using rag_query plus grep over data/content.
always: true
---

# RAG Document Search

Use these tools on the ingested HKU degree programme corpus (curriculum, credit rules, prerequisites, graduate employment statistics, salary figures):

- **grep** with `path="data/content"` — literal text search across copied `.md` files. Use first for exact values: course codes, credit counts, salary figures, verbatim policy text, and other accuracy-sensitive lookups. Set `output_mode='content'` to see matching lines. Use `fixed_strings=true` for exact quotes.
- **rag_query** — semantic + graph retrieval. Use for conceptual, cross-year, fuzzy, or context-heavy questions. Default `mode='mix'`.

**Always use these tools first before searching online for anything about HKU programmes or graduate employment data.**

Use `path="data/content/<subdir>"` on `grep` to narrow to a specific folder. If unsure, start from `data/content`.

For markdown-only searches, add `type="md"` when helpful.

If the task involves calculations, aggregation, or custom analysis over many files, consider writing code to compute the answer instead of relying only on retrieval.

If the data volume is especially large or the work splits naturally into independent chunks, consider using a subagent.

Images in parsed markdown (e.g. `![](images/abc.jpg)`) are relative to the markdown file's directory inside `hybrid_auto/`. Load with `read_file` for visual analysis.
