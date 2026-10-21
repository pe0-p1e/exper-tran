# Conditional cross-model layer allocation: measured development finding

This is Gemma4 E4B held-out development validation on ten screening images, used for selection. It is NOT Gemma31B confirmation and not an independent statistical result. Layer IDs are zero-based.

| Configuration | Active proxies | Total tapped units | Hits/N |
|---|---:|---:|---:|
|InternVL2B[14]|1|1|7/10|
|InternVL2B[11,14]|1|2|7/10|
|CLIP-L14[11]+InternVL2B[11,14]|2|3|9/10|
|CLIP-L14[11]+InternVL2B[11,14,20]|2|4|9/10|
|CLIP-L14+Qwen2B+InternVL2B, four uniform layers/model|3|12|9/10|

Adding Intern11 to Intern14 gains one successful image but loses one; adding CLIP11 to that two-layer proxy gains two and loses none. Adding Intern20 to the resulting three-unit ensemble gains/loses none. CLIP11 alone scored4/10, so it cannot be rejected merely because another singleton scores higher. The actual three-unit condition is a current exploration30 finalist. Beam retained conditional branches; these are actual reoptimized joint images, not a union of singleton successes.

For the three-unit ensemble, the Intern objective is the mean of its layer11 and14 PP losses. Compute one gradient for that mean, normalize per-image L1, then equally fuse with the normalized CLIP11 gradient. Each layer uses the same fixed1:1 cosine prototype PP. Adding layers changes the proxy gradient direction and relative layer contribution; these data do not uniquely identify a semantic mechanism.

The result supports testing conditional model-layer complementarity and stopping redundant additions. It matches the Uniform development hit count with fewer constraints; it does not yet establish higher ASR, statistical equivalence, lower walltime, universal transferability or superiority on Gemma31B. Runtime/backward/VRAM remain inspectable in joint_selection_path.json; concurrent execution/offloading affects measured walltime. Reproduce with scripts/analyze_v8_joint_selection_path.py against persistent full receipts.
