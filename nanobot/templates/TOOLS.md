# Tool Usage Notes

Tool signatures are provided automatically via function calling.
This file documents non-obvious constraints and usage patterns.

## exec — Safety Limits

- Commands have a configurable timeout (default 60s)
- Dangerous commands are blocked (rm -rf, format, dd, shutdown, etc.)
- Output is truncated at 10,000 characters
- `restrictToWorkspace` config can limit file access to the workspace

## glob — File Discovery

- Use `glob` to find files by pattern before falling back to shell commands
- Simple patterns like `*.py` match recursively by filename
- Use `entry_type="dirs"` when you need matching directories instead of files
- Use `head_limit` and `offset` to page through large result sets
- Prefer this over `exec` when you only need file paths

## grep — Content Search

- Use `grep` to search file contents inside the workspace
- For ingested corpus/document searches, start with `path="data/content"`
- For corpus discovery, prefer finding the relevant directories under `data/content` first; only narrow into `hybrid_auto` or other subfolders after you know which programme/year folder you need
- Default behavior returns only matching file paths (`output_mode="files_with_matches"`)
- Supports optional `glob` filtering plus `context_before` / `context_after`
- Supports `type="py"`, `type="ts"`, `type="md"` and similar shorthand filters
- Use `fixed_strings=true` for literal keywords containing regex characters
- Use `output_mode="files_with_matches"` to get only matching file paths
- Use `output_mode="count"` to size a search before reading full matches
- Use `head_limit` and `offset` to page across results
- Prefer this over `exec` for code and history searches
- Binary or oversized files may be skipped to keep results readable

## message — Sending Files to Users

- The **only** way to deliver a file (image, chart, document, audio) to the user is via the
  `message` tool with the `media` parameter containing the absolute file path(s).
- Do NOT embed files as markdown image links (e.g. `![alt](path)`). Markdown links are
  rendered only in text and are never transmitted as actual attachments.
- When `exec` saves a file to a relative path such as `plot.png`, it is written to the
  workspace root. Pass its absolute path to `message`:

  ```python
  message(content="Here is your chart.", media=["/home/nanobot/.nanobot/workspace/plot.png"])
  ```

- Always call `message` with `media` immediately after generating the file — do not wait for
  the user to ask again.

## cron — Scheduled Reminders

- Please refer to cron skill for usage.
