## WEEk 8

This project implements a **rule-based Agentic AI pipeline** that intelligently routes natural language queries to the appropriate tool using keyword and pattern matching. The system analyzes user queries, determines the intended task, invokes the corresponding tool, and returns standardized responses while handling unsupported queries gracefully.

---

## 🚀 Implementations

- ✅ Analyzed the baseline **calculator** and **keyword extractor** tools provided.
- ✅ Built the `agent(query)` function to route user queries based on lowercase keyword matching.
- ✅ Routed **calculation-related** queries to the calculator tool for mathematical evaluation.
- ✅ Routed **keyword extraction** queries to the keyword extractor tool for text processing.
- ✅ Directed all unmatched queries to a general fallback response handler.
- ✅ Detected tool-level failures by validating returned results, since the tools fail silently without raising exceptions.
- ✅ Standardized every response into a consistent JSON format:

```json
{
  "type": "...",
  "result": "..."
}
```

- ✅ Validated the routing pipeline using an automated set of test queries covering all supported response types.
- ✅ Tested the system interactively using a continuous `while True` query loop.
- ✅ Documented observations for each test case, including routing decisions and corresponding outputs.

### ⭐ Bonus Enhancements

- ✅ Improved routing accuracy using **regex-based pattern matching** with keyword synonyms.
- ✅ Added **persistent logging** to record every user query and generated response.
- ✅ Extended the agent by integrating two additional tools:
  - **Word Counter**
  - **Text Reverser**
