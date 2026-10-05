<div>
<div align="right">
<a href="https://piebald.ai"><img width="200" top="20" align="right" src="https://github.com/Piebald-AI/.github/raw/main/Wordmark.svg"></a>
</div>

<div align="left">

### Check out Piebald
We've released **Piebald**, the ultimate agentic AI developer experience. \
Download it and try it out for free!  **https://piebald.ai/**

<a href="https://piebald.ai/discord"><img src="https://img.shields.io/badge/Join%20our%20Discord-5865F2?style=flat&logo=discord&logoColor=white" alt="Join our Discord"></a>
<a href="https://x.com/PiebaldAI"><img src="https://img.shields.io/badge/Follow%20%40PiebaldAI-000000?style=flat&logo=x&logoColor=white" alt="X"></a>

<sub>[**Scroll down for Splitrail.**](#splitrail) :point_down:</sub>

</div>
</div>

<div align="left">
<a href="https://piebald.ai">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://piebald.ai/screenshot-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="https://piebald.ai/screenshot-light.png">
  <img alt="hero" width="800" src="https://piebald.ai/screenshot-light.png">
</picture>
</a>
</div>

# Splitrail

Splitrail is a **fast, cross-platform, real-time token usage tracker and cost monitor for**:
- [Gemini CLI](https://github.com/google-gemini/gemini-cli) (and [Qwen Code](https://github.com/qwenlm/qwen-code))
- [Claude Code](https://github.com/anthropics/claude-code)
- [Codex CLI](https://github.com/openai/codex)
- [Cline](https://github.com/cline/cline) / [Roo Code](https://github.com/RooCodeInc/Roo-Code) / [Zoo Code](https://github.com/Zoo-Code-Org/Zoo-Code/) / [Kilo Code](https://github.com/Kilo-Org/kilocode) (VS Code extension + CLI)
- [GitHub Copilot](https://github.com/features/copilot) (VS Code)
- [GitHub Copilot CLI](https://github.com/features/copilot)
- [OpenCode](https://github.com/sst/opencode)
- [Pi Agent](https://github.com/badlogic/pi-mono/tree/main/packages/coding-agent)
- Grok
- [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)

Run one command to instantly review all of your CLI coding agent usage.  Upload your usage data to your private account on the [Splitrail Cloud](https://splitrail.dev) for safe-keeping and cross-machine usage aggregation.  From the team behind [<img src="https://github.com/Piebald-AI/piebald-issues/raw/main/assets/logo.svg" width="15"> **Piebald.**](https://piebald.ai/)


> [!note]
> ⭐ **If you find Splitrail useful, please consider [starring the repository](https://github.com/Piebald-AI/splitrail) to show your support!** ⭐


**Download the binary for your platform on the [Releases](https://github.com/Piebald-AI/splitrail/releases) page.**

## Screenshots

### [Splitrail CLI](https://splitrail.dev)
<img width="750" alt="Screenshot of the Splitrail CLI" src="https://raw.githubusercontent.com/Piebald-AI/splitrail/main/screenshots/cli.gif" />

### [Splitrail VS Code Extension](https://splitrail.dev)
<img width="750" alt="Screenshot of the Splitrail VS Code Extension" src="https://raw.githubusercontent.com/Piebald-AI/splitrail/main/screenshots/extension.png" />

### [Splitrail Cloud](https://splitrail.dev)
<img width="750" alt="Screenshot of Splitrail Cloud" src="https://raw.githubusercontent.com/Piebald-AI/splitrail/main/screenshots/cloud.png" />

## Local Web Dashboard

Run `splitrail web` and open `http://127.0.0.1:8765`; use `splitrail web --port 9000` to choose another local port. The server only listens on the local loopback address, and no usage data is uploaded. Browser tabs share the latest loaded snapshot; Refresh rescans local usage files, and requests arriving during a scan share its result.

One filter row scopes every view: time range (today, last 7, 30, or 90 days, or all), tool, model, project, and metric (estimated cost or tokens). Type in a tool, model, or project menu to narrow its choices, then select any number of values. Selections within one field are combined; fields narrow each other. The view, filters, and selected session live in the URL hash, so reloading or bookmarking the page keeps them.

- Overview leads with estimated cost compared with the previous period of the same length ("today" compares with yesterday up to the current hour), followed by total tokens with their cached, input, and output mix, sessions, and cache share. A stacked chart shows usage per hour, day, week, or month (weeks start on Monday), grouped by model, tool, or project. Any interval works with any multi-day range; charts too dense for the page scroll sideways and open at the latest period. Empty periods stay visible, and clicking a daily bar drills into that day by hour. Clicking a row in the model, tool, or project ranking adds it to the filters. A weekday × hour heatmap shows when usage happens; hover a cell, or focus the heatmap and use the arrow keys, to see its total, its share of the range, and the other metric. The most expensive sessions show how concentrated spending is. Models whose usage costs $0 are flagged, since their price is probably missing.
- Sessions lists the sessions in range, searchable by name, project, or tool and sortable by recency, cost, tokens, or duration; the selected metric is the emphasized last column. The detail panel covers the whole session: tokens per hour by model, the model split, the token mix, and the session id.
- Breakdown is a table by period, project, model, or tool with cost, total tokens, token types, cache share, average price per million tokens, sessions, and replies. The selected metric leads the columns, followed by each row's share of it, and project, model, and tool rows are sorted by it. Token counts are abbreviated; hover a cell for the exact count. Periods without usage are left out.

Paths that belong to one repository merge into one project:

- subdirectories and linked worktrees of a checkout that still exists, since a worktree's `.git` file names its main repository;
- worktrees under `<repo>/.worktrees/` or `<repo>/.worktree/`, even after they are removed;
- removed worktrees inside a `<repo>-worktrees/` folder whose main checkout sits in that folder or next to it;
- paths sharing the project id an analyzer records where a session starts, such as the Git remote Codex CLI stores, which also joins moved clones and renamed remotes.

A merged project is named after its existing path with the most activity; hover it to see every merged path.

Within the selected time range, the five models, tools, or projects with the highest cost keep their colors while other filters change; the rest are grouped as Other.

Token total uses the same definition as the TUI: input + output + cached tokens. Reasoning tokens are shown separately. Costs are estimated from public API prices.

The page supports Chinese and English, follows the system light or dark theme, and fits narrow screens. It starts in your browser's language when supported, and the language selector remembers your choice in this browser.

Frontend data tests run with `node --test src/web/data.test.mjs`; Node is only needed for these tests, not for building or running Splitrail.

## MCP Server

Splitrail can run as an [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) server, allowing AI assistants to query your usage statistics programmatically.

```bash
splitrail mcp
```

### Available Tools

- `get_daily_stats` - Query usage statistics with date filtering
- `get_model_usage` - Analyze model usage distribution
- `get_cost_breakdown` - Get cost breakdown over a date range
- `get_file_operations` - Get file operation statistics
- `compare_tools` - Compare usage across different AI coding tools
- `list_analyzers` - List available analyzers

### Resources

- `splitrail://summary` - Daily summaries across all dates
- `splitrail://models` - Model usage breakdown

## Configuration

Splitrail stores its configuration at `~/.splitrail.toml`:

```toml
[server]
url = "https://splitrail.dev"
api_token = "your-api-token"

[upload]
auto_upload = false
upload_today_only = false

[formatting]
number_comma = false
number_human = false
locale = "en"
decimal_places = 2

[logging]
level = "warn"
```

## Development

### Windows

On Windows, we use `lld-link.exe` from LLVM to significantly speed up compilation, so you'll need to install it to compile Splitrail.  Example for `winget`:

```shell
winget install --id LLVM.LLVM
```

Then add it to your system PATH:
```cmd
:: Command prompt
setx /M PATH "%PATH%;C:\Program Files\LLVM\bin\"
set "PATH=%PATH%;C:\Program Files\LLVM\bin"
```
or
```pwsh
# PowerShell
setx /M PATH "$env:PATH;C:\Program Files\LLVM\bin\"
$env:PATH = "$env:PATH;C:\Program Files\LLVM\bin\"
```

Then use standard Cargo commands to build and run:

```shell
cargo run
```

### macOS/Linux

Build as normal:
```
cargo run
```


-----

## License

[MIT](https://github.com/Piebald-AI/splitrail/blob/main/LICENSE)

Copyright © 2026 [Piebald LLC](https://piebald.ai).
