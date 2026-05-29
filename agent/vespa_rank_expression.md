<!-- Distilled from github.com/vespaai-playground/skills (schema-authoring +
     rank-features), scoped to the ONE thing this agent actually emits: a Vespa
     first-phase rank EXPRESSION. It names no ranking features beyond the given
     bm25 baseline, so feature discovery stays on the agent (use the docs tools). -->

# Vespa rank-expression reference

You write **only the first-phase expression** — the formula that goes inside
`expression { ... }`. The harness wraps it in a profile that inherits the `bm25`
baseline and auto-declares every `query(...)` weight you reference. You do **not**
write (and cannot use) `rank-profile`, `inputs`, `function`, `second-phase`,
`match-features`, or `rank-properties` blocks — anything beyond the expression is
ignored. The expression may span multiple lines.

## What an expression is

A numeric formula evaluated per matched document; the higher its value, the
higher the document ranks. It combines four things:

- **Rank features** — tokens of the form `feature(field)` or
  `feature(field).output` that read precomputed match/field signals. The only one
  given to you is the baseline `bm25(description)`. Discover any others (their
  names, field arguments, and named outputs) via `search_vespa_docs` / `fetch_url`;
  this reference names none.
- **Query inputs** — `query(name)`, a query-time scalar. Each one you reference is
  auto-declared with default `0.0` and is swept at query time through the `inputs`
  map on `run_train_eval` / `commit_patch` (no redeploy). Use them as tunable
  coefficients, e.g. `query(w_prox) * <feature>`. Names match `[A-Za-z_]\w*`.
- **Numeric literals** — `0.5`, `10`, `2.0`.
- **Operators and math functions** — `+ - * / %`, parentheses, and built-ins:
  `min(x,y)`, `max(x,y)`, `pow(x,y)`, `sqrt(x)`, `log(x)`, `log10(x)`, `exp(x)`,
  `fabs(x)`, `floor(x)`, `ceil(x)`, `sign(x)`, and
  `if(a <cmp> b, then, else)` with comparisons `< <= == != >= >`.

## The searchable field

The corpus has one indexed text field, **`description`** (string, stemmed,
`enable-bm25`). Field-argument features target it, e.g. `bm25(description)`.

## Form to follow

Keep `bm25(description)` as the dominant term and add features as scaled,
query-weighted additions so a small-magnitude feature can't disorder good results:

```
bm25(description)
  + query(w_a) * <some feature>(description)
  + query(w_b) * <another feature>(description).<output>
```

## Pitfalls that cause a zero delta or a failed deploy

- A `query(name)` you reference but never give a nonzero `inputs` value defaults to
  `0.0`, so that whole term contributes nothing — set the weight to see its effect.
- A misspelled or non-existent feature name (or wrong `.output`) fails deployment;
  confirm names against the docs before committing.
- Parentheses must balance and every `feature(...)` needs its field argument.
