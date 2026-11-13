# Workspace file inspection and reading

`inspect-file` analyzes one regular file inside the node workroot. It returns
metadata and a bounded navigation map, without returning the full content.
The entrypoint uses `@deep_tool`; filesystem access, parsing and result selection
live in separate modules under `core/file_inspection`. The snapshot owns
filesystem metadata; the navigation index owns its budget, ranking and pagination,
and exposes immutable selected pages rather than a mutable region list.

## Availability and calls

The `file-processing` skill and its `inspect-file` tool belong to
`DefaultCapabilities(harness=True)`, together with `read-file` and `edit-file`. The bundled main
HDS declares the skill
in `sub-skills`; `inspect-file` is available inside that skill, with parent
tools excluded. Custom HDSs must declare `file-processing` in `sub-skills`
to enter it. The skill provides inspection, text-line and PDF-page reads, direct image reads,
and [interactive text editing](15-interactive-file-editing.md). Individual
selection,
exclusion, named overrides and Before/After hooks follow the existing
default-capability API.

Model arguments:

| Argument | Default | Meaning |
| --- | --- | --- |
| `path` | required | Relative or absolute file path inside the workroot |
| `detailed` | `false` | Include filesystem metadata, text statistics and configured limits |
| `max_items` | `12` | Maximum regions per page, from 1 to 500. The report token budget can reduce the page |
| `offset` | `0` | Zero-based offset in the selected navigation index |
| `query` | `null` | Rank the bounded index by lexical overlap with names/kinds |
| `read_from_line` | `null` | Start the text read plan at this 1-based physical line |

Examples of provider tool-call arguments (the provider protocol remains JSON):

```json
{"path": "src/service.py", "query": "decode user message", "max_items": 10}
{"path": "docs/design.md", "offset": 50, "max_items": 50, "detailed": true}
{"path": "report.pdf"}
```

The runtime also exposes the common `execution_mode` and `wait_seconds`.
Background work uses `control_tool_execution` and `read_tool_output` unchanged.
Results are ordinary structured Python dictionaries; the shared renderer converts
them to typed XML at the presentation boundary. Immediate and saved/background
reads use the same rendering. No independent file-analysis cache is created:
the harness stores tool outputs according to its existing cache policy.

## Analysis

| Format | Result |
| --- | --- |
| Python / stubs | AST class and function index, qualified nested names, ranges including decorators; coding cookies supported; code is never executed |
| JavaScript / JSX (`.js`, `.mjs`, `.cjs`, `.jsx`) | Imports, functions including assigned arrows, classes and methods |
| TypeScript / TSX (`.ts`, `.mts`, `.cts`, `.tsx`) | JavaScript regions plus interfaces, type aliases, enums and namespaces |
| C (`.c`, `.h`) | Includes, macros, function definitions and prototypes, typedefs and named struct/union/enum definitions |
| HTML (`.html`, `.htm`) | Headings, landmarks, elements with IDs and script/style blocks |
| CSS | Selectors and at-rules, including nested rules |
| Markdown | Nested section ranges, ATX/Setext headings, backtick/tilde fences; headings inside fences ignored |
| JSON / YAML | Key paths and inclusive physical line ranges, node counts; values omitted; aliases bounded and cycles reported |
| JSON Lines (`.jsonl`, `.ndjson`) | Physical record lines searchable by top-level keys, record counts and malformed record locations. Blank lines are skipped and values are omitted |
| TOML | Tables and key paths with source line ranges |
| XML / SVG | Element names and line ranges; document types and entities rejected |
| CSV / TSV | Record/column counts, inconsistent widths, delimiter and heuristic header detection; navigation groups 100 records and respects multiline fields |
| ZIP / Office OOXML containers | Member names, counts, declared expanded size, encryption and unsafe-path indicators; no extraction or workbook/document interpretation |
| TAR, gzip/bzip2/xz TAR | Bounded member scan; explicitly partial totals when a budget stops the scan |
| PDF | Page count, encryption, document properties and first-page dimensions in points; no text extraction, rendering or OCR |
| Raster images | Format, dimensions, color mode and animation indicator from headers; no pixel payload, OCR or EXIF export |
| Other Unicode text | Encoding, physical line count and token-sized reads. Detailed reports add text statistics and the whole-file region |

UTF-8 and BOM-marked UTF-16/UTF-32 are supported. Unknown binary/non-Unicode
files return an explicit unsupported status. Other source languages receive
generic text analysis. The bundled Tree-sitter grammars parse source locally.
When a grammar recovers from a syntax error, `syntax_recovered` and a warning
flag the potentially incomplete index while physical lines remain readable.
JSON Lines also keeps physical lines readable when individual records fail to
parse. Its counts reflect the scanned prefix if a resource limit stops analysis.
PDF metadata uses the declared `pypdf` dependency;
encrypted PDFs report encryption and unavailable properties without requesting
passwords or attempting decryption.

Line ranges are inclusive and 1-based. A trailing newline does not invent an
extra line; CR, LF and CRLF count as physical line breaks, while Unicode text
separators do not. Structured parser positions are translated to these same
physical lines, including JSON/YAML strings with Unicode separators. Non-text
files do not claim a physical line count.

`query` searches the complete bounded index before pagination. It prefers exact
names and returns full-term matches when available, with partial matches as a
fallback. Ties keep document order. A query with no matches returns an empty
region list. Names are tokenized across camelCase and snake_case boundaries.
Display labels can be truncated without removing terms from the search index.
The region budget retains regions in document discovery order before ranking
and pagination. An `offset` applies to that ranked index; keep the same query
when paging. `selection.matching_items` counts the matching regions before paging.
Use the returned `next_offset`, since the number of items can vary by page.

Compact reports target 1,600 tokens or the configured output limit, whichever
is lower. The tool counts the rendered XML and reduces the region page as
needed. A tight budget can omit empty fields, default values and redundant
metadata. Read coordinates, warnings and continuation fields are kept. Very
small limits can require exporting the report if its basic fields still exceed
the budget.

### Token-sized text reads

`read_plan` shows up to five consecutive `ranges` per inspection by default.
When the report budget is tight, it keeps the first range and a continuation line.
Each gives `start_index`, `end_index` and `tokens`. The plan prefers nearby
boundaries from the indexed sections and symbols. It starts at the first
matching region on the current query page when it has a line range, and stops at
that region's end. Without a match it starts at line 1 by default. Pass
`next_line` as `read_from_line` with the same query and offset to continue, or
pass a region's `start_line` to plan reads from that location. The region index
still uses `offset` and `max_items`.

`max_tokens` comes from the node's tool-output settings, configured by
`EXMEKH_LIB_TOOL_OUTPUT_MAX_TOKENS` (5000 by default). The plan counts the same
selected text as `read-file`, with LF between physical lines, using the node's
model tokenizer. XML markup and line numbers are outside this budget. Each
suggestion fits the limit for unchanged content, except a single line that
exceeds it, marked `over_limit`. The plan suggests `0/0` for empty text and is
available after complete text analysis. Direct Python callers can pass
`output_settings=ToolOutputSettings(...)`.

## Status and resource limits

`analysis_status` is `complete`, `invalid_format`, `unsupported`, or
`skipped_limit`. A valid inspection may report unavailable/partial analysis
while keeping `ok=true` and metadata. Path, selection and permission errors
return `ok=false` with `error_kind=invalid_file_inspection`. Unexpected internal
failures use the common runtime error handling: both inline and managed tools
report `tool_execution_failed` with `response-type=failed`. Parser dependencies
are loaded at tool execution, so a missing dependency does not break capability
catalog loading. Such failures use the shared `missing_dependency` result
also returned by calculation and web tools.

`selection` distinguishes pagination (`has_more` / `next_offset`) from index
truncation (`index_truncated`) and label truncation (`labels_truncated`).
`indexed_items` counts retained regions, not hypothetical omitted ones.
A TAR with exactly the allowed number of members is complete; its scan is
partial only when a further member is omitted or the expanded-size budget stops
the scan. Warnings explain parser failures and stopped scans; valid text metrics
remain available when a format parser rejects its input.

Budgets live in `config/file_inspection.py`: 8 MiB input, 10,000 navigation
regions, 200 model tokens per label, 1,000 query characters, 64 structural levels,
100,000 YAML node visits, 10,000 archive entries, 64 MiB decompressed TAR input,
2,000 PDF pages and 40 million image pixels. ZIP members are never decompressed.
Oversized files expose metadata only. Changes detected while reading a file
produce a retryable error; each new inspection reads a fresh snapshot.
Image, TAR/compressed TAR and ZIP/container filename dispatch are configured by
`image_extensions`, `archive_extensions` and `zip_extensions`; binary signature
detection still recognizes ZIP and uncompressed TAR content independently.

`max_label_tokens` replaces `max_label_chars`, retaining the default of 200.
The node supplies its model tokenizer for region names and PDF document
properties, in both inline and background execution. Tokenization is a runtime
dependency and is not exposed in the model's tool schema. A truncated region
name sets `selection.labels_truncated`; document properties use the same budget.
Direct Python calls to `inspect_file` require `tokenizer=...`, for example
`inspect_file(work_root, "report.pdf", tokenizer=provider.get_tokenizer())`.

`TokenTextBudget` owns token counting, prefix validation and bounded truncation.
The navigation index uses it only for its region names; the PDF analyzer receives
the document-property budget explicitly. A tokenizer failure propagates as
`TextTokenizationError` and uses normal tool execution failure handling, instead
of marking a valid source file as `invalid_format`. A missing runtime tokenizer
is also an execution dependency failure, not a file-path or selection error.

Deterministic tests create actual files and cover parsers, metadata, limits,
pagination, invalid paths, symlinks, concurrent modification detection, provider
schemas, background retrieval and a complete skill open/inspect/exit cycle
followed by a saved XML read from the parent scope. They do not contact a model provider.

## Explicit reading and prerequisites

`read-file` accepts exactly three required business arguments: `path`,
`start_index`, and `end_index`. For text, both indices must be positive integers; booleans,
strings, floats and null are rejected. Ranges use physical CR/LF lines, are
1-based and inclusive, and must satisfy `start_index <= end_index <= total_lines`.
The tool never supplies implicit indices, clamps an interval or defaults to
the whole file. Results contain `ok`, `path`, `encoding`, the requested indices,
`total_lines`, and `lines` entries with `line` and `text`. Missing/invalid
arguments and formats without readable physical lines produce
descriptive XML errors. Supported images use the separate direct-read path described
below. Missing/unexpected arguments use the shared runtime
`invalid_tool_arguments` result, with the tool's configured correction guidance.
Invalid path values, indices and line ranges use `invalid_file_read`. NUL paths
are rejected before recovery. An unavailable or non-directory workroot is an
explicit inspection error. The inspection byte budget applies to reading too.

An inspected empty text file accepts the explicit range `start_index=0, end_index=0`
and returns `lines=[]`. This successful read authorizes editing an existing empty
file; other ranges on empty files remain errors.

For text and PDF, always inspect first and choose indices from the actual report. The runtime
retains an internal canonical path and XXH3-128 hash of the exact inspected
bytes. It matches actual assistant calls and full, unchanged results in the
current history, then rechecks bounded file snapshots and hashes before decoding
or selecting content. Merely keeping proof in memory or a report in the tool
cache does not authorize reading. Failed, partial or running analyses, summaries,
references, and analyses removed by skill exit or compaction do not qualify.
The prerequisite is frozen at the start of the original call batch; a sibling
inspection whose result the model has not yet seen cannot authorize that batch's
read. A completed background inspection qualifies only after its full result
and the original inspection call are both visible.

Evidence is private runtime metadata, never additional provider arguments or
hash messages, and is isolated per node. It is not exported with cache records
or JSON history: after a runtime restart or importing another node's history,
inspect again. Before/After hooks still apply once per invocation; filtered or
changed inspection reports cannot authorize a read. A read's recovery request
remains internal even if its After hook transforms the returned error or fails.

`FileToolRuntime` owns node-local proof and background output registration;
`FileInspectionHistory` matches that proof against actual history messages.
`FileInspectionRequests` owns scope checks and deduplication. The generic tool
executor only calls their public methods, and no longer contains file-specific
error strings or reconstructs background calls from snapshot fields. Argument
validation, hash matching and decoding/line selection are separate operations.

Reads and automatic inspections are synchronous: the entire prerequisite
transition completes before the next inference. Explicit `inspect-file` calls
retain their existing wait/background protocol. Automatic inspection uses the
same resolver, workroot, hooks, event recording, XML formatter, skill tracking
and output cache. Every synthetic call has a fresh call ID and exactly one real
result. Recovery only appends messages; it does not rewrite any prefix, use
swap/checkpoint/prefill operations, add special `extra_body` arguments, or require
provider changes. Existing skill exit compaction retains its usual behavior.
If the same batch closes the requesting skill or opens another scope, recovery
does not run in the new scope. The original error remains in the saved skill
transcript/output. Keep the skill open until reading succeeds, or inspect again
after entering the appropriate scope. Loaded overrides preserve the original
tool's synchronous policy, just like runtime overrides.

### Successful read

```text
assistant: inspect-file({"path":"notes.md"})
tool:      analysis with total_lines=12
assistant: read-file({"path":"notes.md","start_index":3,"end_index":5})
tool:      XML with numbered lines 3, 4, 5
```

Further reads may reuse that visible report while the file is unchanged; no
inspection is repeated and no hash-check messages are added.

### Image reads

For a filename with a supported image extension, `read-file` delivers the actual
image without requiring inspection. Supply `start_index=0` and `end_index=0`; their
values are ignored, including nonzero, negative or inverted intervals. The three
arguments remain required. The shared `FILE_INSPECTION.image_extensions` config
controls image dispatch for both reading and inspection: `.png`, `.jpg`, `.jpeg`,
`.gif`, `.webp`, `.bmp`, `.tif`, `.tiff`, `.avif`, and `.ico`. Filename extension
matching is case-insensitive; the image decoder validates the actual content.
PDF filenames use inspected page ranges as described below; other filenames
retain the text-read prerequisite and validation.

```text
assistant: read-file({"path":"photo.jpg","start_index":0,"end_index":0})
tool:      XML metadata plus an image attachment
           [next inference receives the image]
```

The XML result contains path, image format, width, height, frame `1`, and zero range
indices. The first frame/page is decoded, EXIF orientation is applied, and pixels
are normalized to RGBA for transport and caching. The source file is closed before
delivery. Both provider APIs receive the multimodal companion through the existing
attachment path, after all replies in the call batch; saved-output reads restore
the image independently of the source file.

The configured byte and pixel limits also apply to image reads. Missing, forbidden,
unreadable, corrupt, undecodable or oversized images return `invalid_file_read`
with an explanatory error. These failures never trigger automatic inspection.
Successful images use `AttachedToolResult`: metadata stays structured for the
shared XML renderer and cache, while image attachments are stored separately.

### PDF page reads

For `.pdf` filenames (case-insensitive), `read-file` requires a complete, visible
inspection of the unchanged file so the model knows its page count. The existing
hash, history and automatic-inspection recovery apply. The required arguments
`start_index` and `end_index` select inclusive **page numbers**, starting at 1,
rather than text lines. Zero, invalid types, inverted ranges and pages outside
the actual document return `invalid_file_read`. Reading more than 5 pages in
one call also returns an error; no range is truncated or clamped.

```text
assistant: inspect-file({"path":"report.pdf"})
tool:      metadata with page_count=12
assistant: read-file({"path":"report.pdf","start_index":4,"end_index":8})
tool:      XML metadata plus five ordered page images, labelled 4 through 8
```

The renderer uses `pypdfium2` directly on the bounded byte snapshot, without
temporary PDF files, external commands, text extraction or OCR. Only the selected
pages are rendered. `FILE_INSPECTION.pdf_render_dpi` defaults to **120 DPI**;
`max_pdf_render_pages` defaults to **5**, independently of the inspection's
2,000-page metadata budget. `max_image_pixels` limits the total rendered pixels
per read and is checked before allocating each page bitmap. The input byte budget
also applies. Native document, page and bitmap handles are explicitly closed;
returned images own their pixels. PDFium calls in this renderer are serialized
across nodes because the native library cannot be called concurrently in threads.
The native renderer is loaded only when a validated, inspection-authorized PDF
read reaches conversion. Missing PDFium therefore does not prevent metadata
inspection or automatic prerequisite recovery. DPI and rendering page limits
must be positive integers; booleans, strings and nonfinite/fractional values
are rejected before renderer loading.

Results include `start_index`, `end_index`, `total_pages`, `rendered_pages`, `dpi`
and page dimensions, with a label and image per selected page. Images reach both
provider APIs and saved-output rereads through the same attachment/cache path.
Invalid, empty, password-protected or unrenderable documents return descriptive
errors; a failed render delivers no partial successful attachment result.
`inspect-file` continues to return PDF metadata only.

### Client integration tests

`tests/core/file_inspection/test_reading_client.py` runs complete simulated-client
flows for text, an image and PDF pages, including skill entry/exit, prerequisites,
real file contents, multimodal delivery and cached output rereads. The same fixtures
are used by three opt-in tests in `tests/live/test_file_reading_live.py`. These
require the configured `BASE_URL`, an image-capable model (`EXMEKH_LIB_LIVE_MODEL`),
and the selected `EXMEKH_LIB_API` (default `responses`). The text token and visual
answers are absent from the prompts, so the real model must use the read results.
The live provider records actual requests and limits each scenario to 12 turns.
Each scenario uses an independent agent context so the server's tool registry
cannot collide with a preceding test or another client session.

```bash
uv run pytest tests/core/file_inspection/test_reading_client.py
uv run pytest -n 0 --run-live tests/live/test_file_reading_live.py
```

### Missing or invalidated analysis

```text
assistant: read-file({"path":"notes.md","start_index":3,"end_index":5})
tool:      XML error_kind=file_inspection_required
assistant: inspect-file({"path":"notes.md"})       [synthetic, new ID]
tool:      real XML analysis of current content
           [next inference]
assistant: read-file({"path":"notes.md","start_index":7,"end_index":9})
tool:      XML with lines chosen by the model from the new analysis
```

Changing even one byte invalidates the old hash. Recovery never executes the
rejected read or reuses its indices; the model must request a new interval.

### Mixed and repeated files

```text
assistant: [read-file(A,1,2), read-file(B,3,4), read-file(C,1,1), read-file(B,5,6)]
tool:      A content (visible analysis valid)
tool:      B prerequisite error
tool:      C prerequisite error
tool:      B prerequisite error
assistant: [inspect-file(B), inspect-file(C)]      [synthetic, fresh IDs]
tool:      B analysis
tool:      C analysis
           [next inference]
```

All original results precede the synthetic group. Resolved aliases of the same
file are deduplicated. Valid reads and unrelated operations are not rerun.

### Automatic inspection failure

```text
assistant: read-file({"path":"missing.txt","start_index":1,"end_index":2})
tool:      XML prerequisite error
assistant: inspect-file({"path":"missing.txt"})   [synthetic, new ID]
tool:      XML path/permission/inspection error
           [next inference; no automatic retry and no content read]
```

The model receives both outcomes and can correct the path or choose another
action. Deterministic provider tests cover these flows, unchanged history
prefixes, call/result pairing, physical-line encodings, hash changes, compaction,
hook handling and node isolation. They require no live model.
