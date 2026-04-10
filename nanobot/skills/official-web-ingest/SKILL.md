---
name: official-web-ingest
description: Collect official website content into the workspace for later grep/RAG/analysis. Use when the user wants the agent to discover pages on an allowed domain, fetch them, convert main content to Markdown, and save raw + cleaned copies plus metadata/index files under `data/`.
metadata: {"nanobot":{"emoji":"🕸️"}}
---

# Official Web Ingest

Use this skill when the user wants a reusable local corpus from official websites rather than a one-off answer.

Typical tasks:
- collect programme pages from an official university site
- turn a list page plus detail pages into local Markdown files
- refresh previously downloaded official web content under `data/`

## Trigger phrases

Use this skill immediately when the user asks to:
- "grab/fetch/download website content into `data/`"
- "build a local corpus from official pages"
- "collect programme pages from a university website"
- "save webpages as markdown for RAG or grep"
- "refresh the official website data"

## Goal

Produce a stable local dataset with:
- raw source copies for traceability
- cleaned Markdown for `grep` and RAG
- metadata/index files for later automation

Prefer repeatable ingestion over ad-hoc browsing.

## Default output layout

Unless the user asks for a different location, write to a dedicated subdirectory under `data/`:

```text
data/<collection>/
  index.json
  raw/
    <slug>.html
  clean/
    <slug>.md
  meta/
    <slug>.json
```

Use a short, stable collection name such as `programmes`, `admissions`, or `careers`.

## Workflow

### Phase 1: Discovery
1. Confirm the allowed source scope from the user request.
   Example: `https://admissions.hku.hk/programmes/undergraduate-programmes` and same-domain detail pages only.
2. Discover candidate pages from the entry page.
   Prefer official list pages and detail pages. Skip login pages, search pages, PDFs already stored elsewhere, and obvious navigation duplicates.

### Phase 2: Probe & Strategy Selection
3. Fetch 1–3 sample pages from different sections of the target site.
4. Inspect the raw content to identify:
   - **Content boundaries**: Where does the main content start and end? (e.g., after breadcrumb, before footer/Apply button)
   - **Noise patterns**: Navigation menus, contact forms, cookie banners, repeated sidebars.
   - **Metadata patterns**: Structured fields like `CODE: 6901`, `FACULTY: Science`, or JSON-LD/Open Graph tags.
5. Choose a cleaning strategy based on the analysis:

| Strategy | When to use | How |
|----------|-------------|-----|
| `boundary` | Consistent page layout with clear start/end markers | Define text/regex markers for content start and end |
| `css-selector` | Known DOM structure (when raw HTML is available) | Extract via CSS selector like `main.article-content` |
| `readability` | Article/blog-style pages | Use readability-like algorithm to isolate main text |
| `heuristic` | Mixed or unknown structure | Remove nav/footer by pattern density, keep text-rich blocks |
| `passthrough` | Already clean or minimal pages | Keep as-is, only remove script/style noise |

6. Define metadata extraction rules:
   - Always extract: `title`, `url`, `fetched_at`, `word_count`, `lang`
   - If structured data exists (JSON-LD, Open Graph), parse it automatically
   - If page-specific fields exist (e.g., programme code, faculty), define regex or positional rules

### Phase 3: Batch Processing
7. Apply the chosen strategy to all remaining pages using a script (Python preferred).
   - Do NOT process pages one-by-one through Agent context — use `exec` to read raw fetch results and write outputs directly.
   - Save raw, clean, and meta files for each page.
   - Update `index.json` incrementally or at the end.
8. Verify a sample of the batch output to ensure the strategy applied correctly.
9. Report what was added, skipped, and any failures.

## Cleaning Strategies

When defining a strategy during Phase 2, document it briefly so it can be reused or adjusted:

```yaml
strategy: boundary
boundaries:
  start: "Undergraduate Courses"  # or regex
  end: "Apply Now" | "loading.gif"
metadata_patterns:
  - key: "CODE"
    pattern: "CODE\\s*\\n+\\s*(\\S+)"
  - key: "FACULTY"
    pattern: "FACULTY\\s*\\n+\\s*(.+)"
```

For sites with varying layouts, consider grouping pages by template type and applying different strategies per group.

## Tool strategy

- Use `web_search` only to discover official pages when the entry page is not enough.
- Use `web_fetch` to read page content. Save results to temporary files for script processing.
- **Prefer `exec` with Python/Shell scripts for bulk processing**, link extraction, cleaning, and index generation. This avoids Agent context膨胀 and handles large datasets reliably.
- Use `write_file` / `edit_file` only for small metadata files or config updates.
- When `web_fetch` results are large, process them via file paths in `.nanobot/tool-results/` rather than reading them back into Agent context.

## Parallel processing

When there are many independent detail pages, prefer sub-agents for faster batch ingestion.

Rules:
- The main agent should first discover, filter, and deduplicate the candidate URLs.
- Use serial execution for small jobs. Switch to sub-agents only when enough independent pages exist to justify batching.
- Split work into disjoint batches so each sub-agent owns a different set of URLs.
- Each sub-agent may write only its assigned `raw/`, `clean/`, and `meta/` page files.
- The main agent must remain the only writer of `index.json` and any shared summary files.
- After sub-agents finish, the main agent validates outputs, merges results, and reports failures.
- If sub-agents are unavailable in the current environment, fall back to serial ingestion without changing the output format.

## Content rules

- Stay on official or explicitly allowed domains.
- Prefer canonical detail pages over announcement pages or mirrors.
- Do not mix unofficial summaries into the corpus unless the user explicitly asks.
- Preserve source traceability. Every cleaned file should point back to its source URL.
- Keep both raw and cleaned forms when possible.
- If a page fetch fails, record the failure instead of silently dropping it.

## Markdown format

For each cleaned Markdown file, add a short frontmatter-style header (unless `passthrough` strategy is used):

```md
---
title: <page title>
url: <source url>
fetched_at: <ISO timestamp>
source_type: official_webpage
---
```

Then include the extracted main content only. Remove obvious navigation, cookie banners, repeated footer text, and script/style noise.

The exact cleaning approach depends on the strategy chosen during Phase 2. Document any site-specific rules in the strategy config.

## Index format

`index.json` should be a JSON array. Each item should include:
- `title`
- `url`
- `slug`
- `raw_path`
- `clean_path`
- `meta_path`
- `fetched_at`

Keep paths relative to the workspace root so later tools can reuse them easily.

## Refresh behavior

When rerunning ingestion:
- reuse existing slugs when the source URL is the same
- overwrite stale raw/clean/meta files for refreshed pages
- preserve unrelated files in the collection
- append clear failure notes for pages that now fail

## HKU programmes pattern

For HKU undergraduate programme ingestion:
- start from the undergraduate programmes list page
- collect programme detail pages on the same official domain
- save them under `data/programmes/`
- prefer one Markdown file per programme page
- include programme code or official page title in metadata when available

## Completion checklist

Before finishing, verify:
- files were actually written under `data/`
- `index.json` points to the created files
- each Markdown file has source URL metadata
- the corpus scope matches the user request
- failures or skipped pages are reported clearly
