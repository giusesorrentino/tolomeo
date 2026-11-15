---
name: critique
description: Review code the user wrote and say whether it is right, why not, and how to see it, when they ask whether their code is correct or ask for a check, without rewriting it for them.
argument-hint: "[file, function or pasted code]"
---

# Critique the user's code

The code to review: $ARGUMENTS (if it is empty, use what the user pasted or the file they mentioned).

Follow the output style `Tutor` (if it is not active, read `.claude/output-styles/tutor.md` first). Do not edit the
user's files and do not paste a corrected version unless asked.

## Procedure

1. **Understand the intent.** Read the user's message, the docstring and the callers. Say in one line what you think
   the code is supposed to do. If you are not sure, say what you assumed.
1. **Read the whole code**, then run it on a small input. Write throwaway scripts in `.tolomeo/scratch/` or use
   `spin test`. Where it is possible, compare the result with a short numpy reference. Use a non-square input, an input
   with NaN, and a fully masked input. If something cannot run, say so.
1. **Report in this order:**
   - **Verdict**, one line: it works, it works with problems, or it gives a wrong result.
   - **What is right**, specific and brief.
   - **Problems**, the most serious first. Group them as wrong results, silent failures (dtype, contiguity, NaN, axis
     order, mask polarity, unit), and consistency with the repository. For each: where it is, what goes wrong, *why*,
     a minimal input that shows it, and the direction of the fix in words.
   - **Style**, at most three remarks, last. ruff and pre-commit handle most of it.
   - **What you ran**, with the numbers, and what you could not verify.
   - **Next step**, in one sentence.

## Checklist for this repository

Axis order `(n_samples, n_det)` and `(ny, nx)`; `True` means good for `flags` and `valid`; `pixels` is `uintp` and used
with `valid`; names `nperseg`, `n_freqs`, `n_cols`, `det_pairs`; `__all__` and `__init__.py` updated; docstring matches
the code and gives shapes and dtypes; error messages include the shapes; a problem the user can fix is a warning issued with
`_warn`, progress is `logger.info`, and `print` appears only in `_cli/main.py`; messages use only variables that are
defined on every path (a `NameError` hides until that path runs); comments are in English; inputs are not modified;
defaults of scientific parameters unchanged; a test exists, or is suggested, for the new behaviour.
