# The `.claude` folder of tolomeo

This folder tells Claude Code how to work in this repository. It is not part of the package.

## What is committed

- `CLAUDE.md`: the project (layout, commands, conventions, results already established). Loaded in every session.
- `rules/python.md` and `rules/tests.md`: rules that load only when Claude touches the files of `src/tolomeo/` or of
  `tests/`.
- `settings.json`: permissions for anyone who uses the repository. The tests (`spin test`, `uv run pytest`) and the
  scripts in `.tolomeo/scratch/` run without asking; `git add`, `git commit`, `git push`, `git reset`, `git clean` and
  `rm` always ask.
- `skills/diagnose-mapmaking`: loads by itself when a map looks wrong, and goes through the checks in order, from the
  pointing to the solver.
- `skills/guide`, `skills/explain`, `skills/critique` and `output-styles/tutor.md`: the tutoring way of working.
  `/guide <what>` walks you through writing a function yourself, `/explain <what>` explains without writing code, and
  `/critique <what>` reviews code you wrote without rewriting it. The output style `Tutor` is in the repository, but it
  is active only for whoever selects it.

## What stays personal (ignored by git)

- `.claude/settings.local.json`: your own settings, added to those of `settings.json`. For example the output style
  (`"outputStyle": "Tutor"`) and a confirmation before every edit in `src/` and `tests/`.
- `CLAUDE.local.md`, at the root of the repository: your own instructions, such as the language of the answers or the
  paths of the data on your computer.

## How to use it

- Open Claude Code in the folder `tolomeo/`. `/status` shows which settings are active, `/context` whether `CLAUDE.md`
  and `CLAUDE.local.md` are loaded.
- To change the output style: `/config`, then *Output style*. The choice is saved in `settings.local.json`, where you
  can also edit `outputStyle` by hand (`"Tutor"`, or remove the line for the default style).
- To let Claude edit files without asking every time, remove the `Edit(...)` lines from `ask` in your
  `settings.local.json`.
- After editing an output style or a skill, restart Claude Code: they are read at startup.

## Pre-commit

This folder is excluded from the `mdformat` hook of `.pre-commit-config.yaml` (`exclude: ^\.claude/`) and must stay
excluded. Without the plugin `mdformat-frontmatter`, mdformat turns the frontmatter (the lines between `---`) of styles,
rules and skills into a horizontal rule and a heading: Claude Code no longer reads them, and no error says so.
