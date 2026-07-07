## Information Priority

When answering any question about the project, follow this order:
1. **Obsidian** (`D:\Obsidian-knowledgebase\AI-knowledge-base`) — check for relevant notes first
2. **Conversation context** — what was discussed in this thread
3. **Code** (`C:\Users\Curtis\Desktop\learningmaterials\SEMESTER3\bakery-ai-system`) — read source as last resort

Do not assume or guess. Check Obsidian before writing code or making decisions.


## Coding Rules (from Obsidian hermes/code-rules.md)

### No Hardcoded Values
- Array sizes must be dynamic, never fixed numbers
- Hours/ranges must come from DB queries (MIN/MAX), never hardcoded
- Chart options (width, rotation, symbolSize, grid) must adapt to data count/length
- Thresholds must derive from actual data, not magic numbers

### No Fabricated Defaults  
- Empty data = query a meaningful fallback range from DB
- Missing values = NULL or skip, never `or 0` / `or 8` / `or 21`
- No fake payment data, no mock chart data to "make it look good"
- All dashboard numbers from real DB queries only

### Data Sources
- Revenue/Profit: orders + order_items (MySQL)
- Payment methods: payments table, COUNT not SUM
- Products: products table
- Never seed/simulate/mock data for display

### Standards
- English only in code
- 127.0.0.1, never localhost
- Currency: CNY
- Server: Uvicorn
- No _core.js
- Prefer standard library and built-in platform APIs whenever they meet the requirement; add third-party dependencies only when they provide clear, necessary value.
- Do not use PowerShell as the primary way to write or modify code. Use `apply_patch` for manual edits and reserve shell commands for inspection, testing, formatting, and running tools.

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **git** (16079 symbols, 19375 relationships, 194 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> Index stale? Run `node .gitnexus/run.cjs analyze` from the project root — it auto-selects an available runner. No `.gitnexus/run.cjs` yet? `npx gitnexus analyze` (npm 11 crash → `npm i -g gitnexus`; #1939).

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows. For regression review, compare against the default branch: `detect_changes({scope: "compare", base_ref: "main"})`.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `query({search_query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `context({name: "symbolName"})`.
- For security review, `explain({target: "fileOrSymbol"})` lists taint findings (source→sink flows; needs `analyze --pdg`).

## Never Do

- NEVER edit a function, class, or method without first running `impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `rename` which understands the call graph.
- NEVER commit changes without running `detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/git/context` | Codebase overview, check index freshness |
| `gitnexus://repo/git/clusters` | All functional areas |
| `gitnexus://repo/git/processes` | All execution flows |
| `gitnexus://repo/git/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
