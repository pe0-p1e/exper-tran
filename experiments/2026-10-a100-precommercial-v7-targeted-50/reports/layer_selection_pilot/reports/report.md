# Small layer/model selection pilot

Ten predeclared conditions, same historical development images, Qwen9 held-out cross-family surrogate only. Fixed balanced Pull/Push, DI and equal per-image L1-normalized fusion. Six layer conditions retain E11 model order; four model-count controls retain four uniform layers. 50 steps,48 references/class, strict BF16. Not formal target results; selection bias and n=10 limit interpretation. The four-uniform matched baseline is reused only after source, cohort and PNG integrity checks. Non-quartile taps are exploratory controls, not a reproduction of ILA latest-peak or FDA greedy classifier selection.

single37p5: COMPLETE 1/10
single62p5: COMPLETE 5/10
single87p5: COMPLETE 7/10
single75: COMPLETE 7/10
two62p5_87p5: COMPLETE 7/10
four_uniform: COMPLETE 9/10
Intern_single: COMPLETE 7/10
Gemma_single: COMPLETE 4/10
CLIP_single: COMPLETE 7/10
Intern_Gemma_pair: COMPLETE 7/10
