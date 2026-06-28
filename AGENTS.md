
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
