# Cursor Tools Usage Guide

This document provides a comprehensive guide to all the tools available in Cursor and their proper usage patterns.

## Core Principles

### 1. Parallel Tool Execution
**CRITICAL**: Always execute multiple tools simultaneously rather than sequentially when possible. This is 3-5x faster and significantly improves user experience.

- **DEFAULT TO PARALLEL**: Unless output of Tool A is required for input of Tool B, execute tools simultaneously
- **Information Gathering**: When you need multiple pieces of information, plan all searches upfront and execute together
- **File Operations**: Reading multiple files, searching different patterns, or exploring different directories should all happen in parallel

### 2. Maximize Context Understanding
- **Be THOROUGH**: Get the FULL picture before replying
- **TRACE symbols**: Follow every symbol back to definitions and usages
- **EXPLORE comprehensively**: Look past first results, use varied search terms
- **Semantic search first**: Start broad with high-level queries, then narrow down

### 3. Code Changes Philosophy
- **NEVER output code** to user unless requested - use edit tools instead
- **Prefer editing existing files** over creating new ones
- **Make code immediately runnable**: Add all imports, dependencies, endpoints
- **Fix linter errors** if clear how to (max 3 attempts per file)

## Tool Categories

## File System Tools

### `read_file`
**Purpose**: Read file contents from filesystem
**Usage**:
- Can specify line offset and limit for large files
- Lines numbered starting at 1 (LINE_NUMBER|LINE_CONTENT format)
- Use parallel calls to read multiple files simultaneously
- Always better to read multiple potentially useful files as a batch

```
read_file(target_file="path/to/file.py", offset=10, limit=20)
```

### `write`
**Purpose**: Write/overwrite files
**Critical Rules**:
- MUST read existing file first if it exists
- ALWAYS prefer editing existing files over creating new ones
- NEVER proactively create documentation files unless explicitly requested
- Will overwrite existing files

### `search_replace`
**Purpose**: Exact string replacements in files
**Key Points**:
- Preserve exact indentation (tabs/spaces)
- Edit will FAIL if old_string is not unique - provide more context
- Use `replace_all=true` for renaming variables across file
- Prefer editing existing files in codebase

### `MultiEdit`
**Purpose**: Multiple edits to single file in one operation
**Best Practices**:
- Built on search_replace, allows multiple find-replace operations
- All edits applied sequentially in order provided
- Atomic operation - either all succeed or none applied
- Use when making several changes to different parts of same file

### `list_dir`
**Purpose**: List files and directories
**Features**:
- Can use relative or absolute paths
- Optional glob patterns to ignore files
- Does not display dot-files/directories

### `glob_file_search`
**Purpose**: Find files by name patterns
**Usage**:
- Fast with any codebase size
- Returns paths sorted by modification time
- Patterns auto-prepended with `**/` for recursive search
- Use for finding files by name patterns

## Search Tools

### `codebase_search` 
**Purpose**: Semantic search for code by meaning, not exact text
**When to Use**:
- Explore unfamiliar codebases
- "How/where/what" questions about behavior
- Find code by meaning rather than exact text

**When NOT to Use**:
- Exact text matches (use `grep`)
- Reading known files (use `read_file`)
- Simple symbol lookups (use `grep`)
- Finding files by name (use `glob_file_search`)

**Best Practices**:
- Start with broad, exploratory queries
- Use complete questions: "How does user authentication work?"
- Avoid vague single words: "AuthService" → "What is AuthService used for?"
- Break large questions into smaller focused sub-queries
- Target directories: provide ONE directory, [] for everywhere
- Run multiple searches with different wording

**Search Strategy**:
1. Start exploratory with [] if unsure where code is
2. Review results, narrow to specific directories if needed
3. For large files (>1K lines), use semantic search rather than reading entire file

### `grep`
**Purpose**: Powerful search built on ripgrep
**When to Use**:
- Exact symbol/string searches
- Full regex syntax support
- When you know specific text you're looking for

**Key Features**:
- Respects .gitignore/.cursorignore
- Output modes: "content" (default), "files_with_matches", "count"
- Context lines: -A (after), -B (before), -C (both)
- Case insensitive: -i flag
- Multiline matching: multiline=true
- File type filtering: type parameter or glob patterns

**Best Practices**:
- Prefer over terminal grep/rg
- Escape special chars for exact matches: `functionCall\\(`
- Use `type` or `glob` when certain of file type needed
- Results capped for responsiveness

## Development Tools

### `run_terminal_cmd`
**Purpose**: Execute terminal commands
**Critical Rules**:
- Commands require user approval
- If new shell: cd to appropriate directory and setup
- If same shell: check chat history for current working directory
- NON-INTERACTIVE FLAGS: Pass --yes for commands requiring interaction
- PAGER COMMANDS: Append ` | cat`
- LONG-RUNNING: Set `is_background=true` instead of modifying command
- NO NEWLINES in command

### `read_lints`
**Purpose**: Read linter errors from workspace
**Usage**:
- Provide specific file/directory paths or omit for all files
- Can return pre-existing errors before your edits
- NEVER call on files unless you've edited them or are about to
- Avoid very wide scope

### `delete_file`
**Purpose**: Delete files safely
**Features**:
- Fails gracefully if file doesn't exist
- Security protections built-in
- Clean up temporary files created during iteration

## Specialized Tools

### `edit_notebook`
**Purpose**: Edit Jupyter notebook cells
**Key Rules**:
- ONLY tool for editing notebooks
- Cell indices are 0-based
- `is_new_cell`: true for new cells, false for editing existing
- Must provide ALL required arguments including both old_string and new_string
- old_string must uniquely identify instance with 3-5 lines context before/after
- Can only change ONE instance at a time
- May save markdown cells as "raw" - this is fine

### `fetch_pull_request`
**Purpose**: Get PR/issue details by number or commit by hash
**Usage**:
- Returns full diff and metadata
- Prefer over manual git commands
- Useful for understanding recent changes
- Link to PRs/issues in markdown: `[PR #123](https://github.com/org/repo/pull/123)`

### `create_diagram`
**Purpose**: Create Mermaid diagrams
**Requirements**:
- Use `<br/>` for line breaks
- Wrap text/tags in double quotes
- No custom colors, no `:::`, no beta features
- No remote images (security restriction)

### `web_search`
**Purpose**: Search web for real-time information
**Use Cases**:
- Up-to-date information not in training data
- Verify current facts
- Current events, technology updates
- Include version numbers/dates for technical queries

### `update_memory`
**Purpose**: Manage persistent knowledge base
**Actions**:
- `create`: Remember new information (only when user explicitly asks)
- `update`: Modify existing memory (requires existing_knowledge_id)
- `delete`: Remove contradicted memory (requires existing_knowledge_id)
**Rules**:
- NEVER create without explicit user request
- MUST cite memories when used: `[[memory:MEMORY_ID]]`
- Update/delete when user contradicts or information changes

### `todo_write`
**Purpose**: Manage structured task lists
**When to Use**:
- Complex multi-step tasks (3+ steps)
- Non-trivial tasks requiring planning
- User provides multiple tasks
- After receiving new instructions (capture as todos)
- After completing tasks (mark complete, add follow-ups)

**When NOT to Use**:
- Single straightforward tasks
- Trivial tasks (<3 steps)
- Purely conversational requests
- NEVER include: linting, testing, searching codebase

**Task Management**:
- States: pending, in_progress, completed, cancelled
- Only ONE task in_progress at a time
- Mark complete IMMEDIATELY after finishing
- Update status in real-time
- Batch todo updates with other tool calls

## Tool Usage Patterns

### Information Gathering Strategy
1. **Plan searches upfront**: What information do I need?
2. **Execute all searches in parallel**: Don't wait for results before planning next search
3. **Use semantic search for exploration**: Start broad, then narrow
4. **Use grep for exact matches**: When you know specific text
5. **Read multiple files simultaneously**: Batch file operations

### Code Editing Workflow
1. **Read files first**: Understand context before editing
2. **Use MultiEdit for multiple changes**: Single file, multiple edits
3. **Preserve exact formatting**: Indentation, whitespace matter
4. **Fix linter errors iteratively**: One tool at a time, test until zero errors
5. **Clean up temporary files**: Delete when done

### Error Handling
1. **If edit fails**: Read file again before retrying
2. **If old_string not unique**: Add more context around the change
3. **Linter error workflow**: Fix all errors from one tool, test, move to next tool
4. **Maximum 3 attempts**: For linter fixes on same file

### Performance Optimization
- **Default to parallel tool calls**
- **Batch read operations**
- **Use appropriate search tools**: Semantic vs exact match
- **Target specific directories**: When you know the area
- **Read multiple potentially useful files**: Better to over-read than under-read

## Common Mistakes to Avoid

1. **Sequential tool calls**: When parallel would work
2. **Single-word semantic searches**: Use complete questions
3. **Creating new files unnecessarily**: Prefer editing existing
4. **Not reading existing files**: Before writing/editing
5. **Outputting code to user**: Use edit tools instead
6. **Vague search queries**: Be specific about what you're looking for
7. **Not using todo_write**: For complex multi-step tasks
8. **Not citing memories**: When using remembered information
9. **Ignoring linter errors**: Fix iteratively until clean
10. **Not cleaning up**: Remove temporary files when done

## Quick Reference

### File Operations
- Read: `read_file` (batch multiple files)
- Write: `write` (read existing first)
- Edit: `search_replace` or `MultiEdit`
- Find: `glob_file_search` (by name) or `codebase_search` (by meaning)

### Search Operations
- Semantic: `codebase_search` ("How does X work?")
- Exact: `grep` (specific strings/symbols)
- Files: `glob_file_search` (name patterns)

### Development
- Commands: `run_terminal_cmd` (non-interactive flags)
- Errors: `read_lints` (files you've edited)
- Tasks: `todo_write` (complex multi-step work)

### Specialized
- Notebooks: `edit_notebook` (only for .ipynb)
- PRs/Issues: `fetch_pull_request` (by number)
- Diagrams: `create_diagram` (Mermaid)
- Web: `web_search` (current information)
- Memory: `update_memory` (when user asks to remember)

Remember: **Parallel execution is the default**. Only use sequential calls when the output of one tool is genuinely required for the input of the next tool.
