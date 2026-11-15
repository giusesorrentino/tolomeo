---
name: explain
description: Explain in depth a concept, equation, algorithm, API or piece of the code, without writing implementation code, when the user asks only for an explanation or asks for no code.
argument-hint: "[what to explain]"
---

# Explain, without writing the code

The request: $ARGUMENTS (if it is empty, use what the user said in the conversation).

Write in full sentences and follow the output style `Tutor` (if it is not active, read `.claude/output-styles/tutor.md`
first). No implementation code. Snippets are allowed only when they show a tool or a trap on a toy input (at most about
eight lines).

## Structure of the explanation

1. **The question in one sentence**, as you understood it. If it can mean two different things, answer the more likely
   one and say which one you chose.
1. **The idea in words.** What problem the thing solves and why it is needed, with a picture or an analogy when it
   helps.
1. **The mathematics.** The formula, with every symbol defined and its shape. Use the notation of `CLAUDE.md`
   (`d = P m + n`, `A = Pᵀ N⁻¹ P`, `b = Pᵀ N⁻¹ d`, `n_samples`, `n_det`, `nperseg`) so that the user sees the same
   symbols everywhere.
1. **A small numeric example**, worked out by hand with numbers that can be followed, then confirmed by running it
   (in `.tolomeo/scratch/`). Report what you ran.
1. **Where it lives in the code.** The file, the function and the line of reasoning that connects the formula to it.
1. **What is easy to get wrong.** Two or three misunderstandings or traps, specific to this topic.
1. **What to look at next**, in one sentence.

## Calibration

Start from what the question shows the user knows. If the topic is new for them, begin one level lower than you
would by instinct. If it is a topic they use every day, skip the idea and go to the traps.

If the explanation is something the user will want to read again, offer to write it down where they want it (for
example in the docstring of the module); do not edit any file unless they say yes.
