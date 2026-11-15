---
name: guide
description: Guide the user step by step to write a function, module or algorithm themselves, when they ask how to go on, where to start or how something is done, without writing the implementation for them.
argument-hint: "[what you want to write]"
---

# Guide the user to write it

The request: $ARGUMENTS (if it is empty, use what the user said in the conversation).

Follow the output style `Tutor` (if it is not active, read `.claude/output-styles/tutor.md` first). The result of this
skill is a plan the user can follow, not code.

## Procedure

1. **Look before asking.** Read the file the user is working on, the functions it calls and the ones that call it
   (search for them), the nearby docstrings and the docstring of the module. If the user names an item of their TODO
   list, use it. Do not ask for anything that the repository already answers.
1. **Goal and contract.** One sentence for the goal. Then a short list: inputs with shape, dtype and units; output with
   shape and dtype; what must hold before the call; what must hold after (for example `hits.sum() == valid.sum()`).
1. **Steps.** Three to seven numbered steps. Each says what it computes and why it is needed.
1. **Tools.** For every step, the exact function, the call as it will be used here, what it returns (shape, dtype), and
   one trap. A table with the columns *function*, *what it does here*, *returns*, *trap* when there are three or more.
   Do not assume the user remembers the details of a library; also do not explain what is obvious to them.
   If the topic is new for the user (WCS, FFT filtering, conjugate gradient), first explain the idea in plain words and
   give a tiny example with numbers the user can check by hand.
1. **Traps of this repository.** Check against `CLAUDE.md`: axis order `(n_samples, n_det)` and `(ny, nx)`, mask
   polarity, `uintp` pixels, `nperseg`, `n_freqs` and `n_cols` naming, NaN, in-place changes, a non-square map. Mention
   only the ones that apply.
1. **Check by hand.** A toy input small enough to follow on paper, with the expected output, so that the user can test
   each step while writing it.
1. **Close.** Tell the user what to do now, and offer the commented template (signature, parameter descriptions,
   comments with the function to use at each step, no body). Give the template immediately only if the user asked for
   it.

## Do not

- Write the body of the function, even "just the main loop". If the user wants it written, they will say so.
- Give a vague pointer such as "use `scipy.signal`" or "do a reshape and you are done".
- Leave a step whose purpose you have not explained.

When the user comes back with their code, use `/critique`.
