# Prompt Inputs Log — GPT From Scratch

Verbatim record of the prompts that drove this project, per the standing
practice of logging raw inputs for every project going forward.

---

**2026-07-20** — "actually maybe we want to make an entirely new project not
related to the tech and econ we have worked on before we can build more
stuff maybe with CS or machine learning or what do you think do some
research"

**2026-07-20** — "ok lets go with your recommendation and you can go ahead
and spearhead this project with what you deem necessary"

**2026-07-22** — "GPT From scratch on teh desktop" (in answer to "which new
project do you mean?" after a prior session's threads were resolved/deferred)

**2026-07-22** — "Improve quality" (in answer to "where do you want to take
it next?" — chose bigger model / longer training over BPE tokenizer or
KV-caching as the next milestone)

**2026-07-22** — "what are you adding and editing that will take an hour"
(prompted a breakdown of the v2 changes and their compute cost)

**2026-07-22** — "Okay maybe we can break it down into smaller chunks and
map out a little plan on creating this new GPT from scratch model" — led to
a 4-stage plan (isolate LR-schedule/grad-clip effect → isolate model-size
effect → train longer if still improving → compare & decide), each stage its
own checkpoint, checking in between stages instead of one blind hour-long run.
