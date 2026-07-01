# CONTRIBUTING.md — ChatLedger

---

## Before you start

Read `SKILL.md` and `AGENTS.md` first. They document the project conventions,
the non-negotiable constraints, and where things should live. This file covers
the workflow — how to branch, commit, test, and submit changes.

---

## Local setup

```bash
git clone https://github.com/yourname/chatledger.git
cd chatledger
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt   # black, ruff, mypy, bandit, pytest

ollama pull phi3:mini
ollama serve
```

Confirm everything works:
```bash
pytest tests/ -v                   # all tests pass
uvicorn backend.main:app --reload  # server starts, /health returns ok
```

---

## Branching

Use short, descriptive branch names prefixed with the type of change:

```
feat/csv-export
fix/multiline-message-parser
chore/update-ruff-config
docs/update-skill-md
test/extractor-retry-coverage
```

Branch from `main`. One concern per branch. Don't combine a feature and a
refactor in the same branch.

---

## Commit messages

This project uses [Conventional Commits](https://www.conventionalcommits.org/).
The CI commit-lint job enforces this — non-conforming commit messages will
fail the check.

### Format

```
<type>(<scope>): <short description>

[optional body]

[optional footer]
```

### Types

| Type | When to use |
|---|---|
| `feat` | New feature or capability |
| `fix` | Bug fix |
| `chore` | Maintenance, deps, config changes |
| `docs` | Documentation only |
| `test` | Adding or fixing tests |
| `refactor` | Code change with no behavior change |
| `perf` | Performance improvement |
| `ci` | CI pipeline changes |

### Scope (optional but recommended)

Use the module name: `parser`, `extractor`, `merge`, `db`, `schema`,
`frontend`, `ci`, `deps`.

### Examples

```
feat(extractor): add chunk-level cache using sha256 hash key
fix(parser): handle colons inside message text without breaking sender split
test(extractor): add explicit coverage for double-failure fallback path
docs: update SKILL.md with llama3.2:3b performance notes
chore(deps): bump pydantic to 2.7.0
```

### What not to do

```
# Too vague
git commit -m "fix bug"
git commit -m "updates"
git commit -m "WIP"

# Wrong type
git commit -m "feat: fix the parser crash"   # crashes are fixes, not features

# Missing scope on ambiguous messages
git commit -m "fix: handle edge case"        # which module? add scope
```

---

## Before opening a PR

Run all of these locally and fix any failures before pushing:

```bash
black .                        # auto-format
ruff check . --fix             # auto-fix lint issues
mypy backend/                  # type check
bandit -r backend/             # security scan
pytest tests/ -v               # full test suite
```

If `mypy` or `bandit` flag something you believe is a false positive, add an
inline ignore comment with a brief explanation — don't just suppress it blindly:

```python
result = some_func()  # type: ignore[union-attr]  # safe: None case handled above
```

---

## Pull request checklist

Before requesting review, confirm:

- [ ] Branch is up to date with `main`
- [ ] All CI checks pass (green on GitHub Actions)
- [ ] New functionality has corresponding tests
- [ ] No new external HTTP calls introduced
- [ ] No Pydantic validators weakened
- [ ] `SKILL.md` updated if you changed defaults, conventions, or known behaviors
- [ ] `AGENTS.md` updated if you added a new module or changed pipeline structure
- [ ] PR description explains what changed and why, not just what

---

## PR description template

```markdown
## What this changes
[One paragraph explaining the change]

## Why
[What problem this solves or what it enables]

## How to test it manually
[Steps to reproduce or verify the change locally]

## Checklist
- [ ] Tests added/updated
- [ ] CI passing
- [ ] SKILL.md updated if needed
- [ ] No external network calls introduced
```

---

## What gets reviewed

- **Correctness** — does it do what it says? does it handle failure cases?
- **Pipeline integrity** — does it respect the stage boundaries (parse → chunk → extract → merge → store)?
- **Offline constraint** — does it introduce any external network dependency?
- **Test coverage** — is the new behavior actually tested, including failure paths?
- **Schema integrity** — does it maintain or strengthen (not weaken) Pydantic validation?

---

## What will be asked to change

- Any weakening of `schema.py` validators
- Tests that mock at the wrong layer (e.g. mocking `parse_chat` to avoid writing real test data)
- Frontend changes that add CDN dependencies
- Commits that mix unrelated concerns
- Code that calls `extract_chunk()` outside of `extractor.py`
- Any introduced behavior that requires network access

---

## Running a specific test file

```bash
pytest tests/test_parser.py -v
pytest tests/test_extractor.py -v -k "retry"   # run only retry-related tests
```

---

## Reporting a bug

Open a GitHub issue with:
- Python version and OS
- Ollama model in use (`CHATLEDGER_MODEL` or default)
- The exact error message or unexpected behavior
- A minimal sample `.txt` snippet that reproduces it (anonymize real names)

---

## Questions

Check `SKILL.md` first — it covers most "why does this work this way" questions.
If it's not there, open a Discussion rather than an Issue.