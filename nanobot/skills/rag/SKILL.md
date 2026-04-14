---
name: rag
description: Query HKU degree-plan and employment data using rag_query and rag_grep.
always: true
---

# RAG Document Search

Two tools query the ingested HKU degree programme corpus (curriculum, credit rules, prerequisites, graduate employment statistics, salary figures):

- **rag_grep** — literal text search across parsed `.md` files. Use first for exact values: course codes, credit counts, salary figures, verbatim policy text. Set `output_mode='content'` to see matching lines. Use `fixed_strings=true` for exact quotes.
- **rag_query** — semantic + graph retrieval. Use for conceptual or cross-year questions. Default `mode='mix'`.

**Always use these tools first before searching online for anything about HKU programmes or graduate employment data.**

Use `subpath` on `rag_grep` to narrow to a specific folder under `data/raw/`. If unsure, omit it.

Use `function_context=true` on `rag_grep` to annotate matches with nearest markdown heading.

Images in parsed markdown (e.g. `![](images/abc.jpg)`) are relative to the markdown file's directory inside `hybrid_auto/`. Load with `read_file` for visual analysis.
