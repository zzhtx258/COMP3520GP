---
name: rag
description: Query HKU degree-plan and employment data using rag_query plus grep over data/content.
always: true
---

# RAG Document Search

**Always query these tools before searching online for anything about HKU programmes or graduate employment data.**

## Tools

- **rag_query** — semantic + knowledge-graph retrieval. Use this to identify candidate files, sections, entity aliases, and terminology when the question is open-ended, fuzzy, or requires cross-document scoping. It is a scoping tool, not an exact counter.
- **grep** with `path="data/content"` — literal text search. Use this to confirm exact values: course codes, credit counts, salary figures, verbatim policy text. Set `output_mode='content'` to see matching lines; `fixed_strings=true` for verbatim quotes.

Use `path="data/content/<subdir>"` to narrow grep to a specific folder. For markdown-only searches, add `type="md"`.

To calibrate `top_k` for rag_query, check corpus size first with `grep(output_mode="count", pattern=".", path="data/content", glob="*.md")` and scale accordingly.

## Step 1 — Classify the query (easy vs. hard)

Before retrieving anything, decide whether the question is **easy** or **hard**.

**Easy**: the answer can be produced from a small number of directly retrieved passages.
Examples: "What are the credit requirements for COMP3234?", "Which investment banks appeared in 2023?"

**Hard**: the answer requires collecting and aggregating evidence across many independent slices — years, degrees, or entity categories.
Examples: "How many CS graduates entered investment banks each year 2017–2023?", "What is the maximum salary per degree 2017–2023?"

If cross-slice counting, comparison, or trend assembly is required, treat the question as hard.

## Step 2 — Retrieval strategy

**Easy question:**
1. Use grep for exact-match lookups, or rag_query for open-ended entity discovery.
2. Verify with targeted follow-up if needed.
3. Answer directly.

**Hard question:**
1. Call rag_query once with a broad scoping query to identify: relevant years, degree names, file paths, section headings, entity aliases.
2. rag_query output is for scoping only — do not read its numbers as exact counts or final answers.
3. Use the scoping result to narrow subsequent grep / read_file calls to specific candidates.
4. Do NOT open-endedly scan the whole corpus with grep or exec before scoping with rag_query first.
5. For aggregations too large for manual reading, consider writing and running code over the narrowed candidate files instead of reasoning from retrieval alone.

## Step 3 — Optional subagents for independent slices

After scoping a hard question, decide whether subagents are needed. For multi-file or multi-group tasks, consider using a subagent when the work can be split into independent slices. Launch subagents **only** when:
- the question is hard, AND
- the slices are truly independent and have little dependency on each other

Do not launch subagents for easy questions. Do not launch subagents when slices are not independent.

Each subagent task must specify:
- the original user question
- one assigned slice only (a single independent partition)
- the candidate files / sections / aliases from the parent's scoping result
- the subagent must produce explicit evidence for that slice only — NOT a global summary

If an assigned slice is still too large but remains separable, the subagent may further split it into smaller independent sub-slices. Keep delegation bounded and preserve explicit evidence chains from child slices back to parent slice outputs.

## Step 4 — Assemble evidence before answering hard questions

For any hard question, build this compact evidence table before producing the final answer:

| slice_key | matched entities | normalized label | count / value | source | status |
|-----------|-----------------|-----------------|---------------|--------|--------|

Status must be: `ok` (fully verified), `partial` (some gaps), or `failed` (could not verify).

Charts and numeric totals must come from this evidence table — not from narrative inference or rag_query output alone.

## Step 5 — Final answer

**Easy questions:** answer directly.

**Hard questions:**
- Lead with the grounded result.
- Claim only what is supported by the evidence table.
- If any slice has status `partial` or `failed`, explicitly state the result is **incomplete** and identify the missing slices.
- Do not add macro-level narrative explanations unless separately supported by retrieved evidence.
- Output this status line on its own line immediately before the answer text: `[hard-query: subagents=yes|no, answer=complete|partial]`

## Images

Images in parsed markdown (e.g. `![](images/abc.jpg)`) are relative to the markdown file's directory inside `hybrid_auto/`. Load with `read_file` for visual analysis.
