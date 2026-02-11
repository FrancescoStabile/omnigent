# Complexity Analysis Cheatsheet

## Cyclomatic Complexity Thresholds

| CC Score | Risk Level | Action |
|----------|-----------|--------|
| 1-5      | Low       | Well-structured, no action needed |
| 6-10     | Moderate  | Consider simplification |
| 11-20    | High      | Refactoring recommended |
| 21-50    | Very High | Must refactor, split into smaller functions |
| 50+      | Critical  | Untestable — immediate decomposition required |

## Quick Wins

### Extract Method Pattern
When CC > 10, look for:
- Nested if/else chains → extract to named methods
- Loop bodies > 10 lines → extract to method
- Switch/match with > 5 cases → use dict dispatch or strategy pattern

### Guard Clause Pattern
Replace deep nesting:
```python
# Before (CC=5)
def process(data):
    if data:
        if data.valid:
            if data.type == "A":
                return handle_a(data)
            else:
                return handle_b(data)

# After (CC=3)  
def process(data):
    if not data: return None
    if not data.valid: return None
    return handle_a(data) if data.type == "A" else handle_b(data)
```

## Metrics Reference

- **Halstead Complexity**: Based on operators and operands
- **Maintainability Index**: MI = 171 - 5.2*ln(V) - 0.23*G - 16.2*ln(LOC)
  - > 85: Good | 65-85: Moderate | < 65: Poor
- **Cognitive Complexity**: Weights nested control flow higher than flat
