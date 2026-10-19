# Frozen-method commercial version comparison

Stage: dev10; same fixed 10 images per model; clean/adversarial in fresh separate sessions.
Invalid predictions remain misses. Missing access/transport observations are N/A. All parameters/method/images are frozen before requests.
Model IDs are explicit requested client IDs; backend serving snapshots and closed weights are not exposed.
CLI routes and prompts are logged; different clients are not claimed to have identical system context.

| Model | Route | Status | Requests | Target hits / N | ASR |
|---|---|---|---|---|---|
| gpt-6.1-sol | codex | COMPLETE | 20/20 | 4/10 | 0.4 |
| gpt-6-sol | codex | COMPLETE | 20/20 | 8/10 | 0.8 |
| gpt-5.6-sol | codex | BLOCKED | 6/20 | N/A/10 | N/A |
| gpt-6-astra | codex | COMPLETE | 20/20 | 5/10 | 0.5 |
| gpt-6-luna | codex | COMPLETE | 20/20 | 6/10 | 0.6 |
| gpt-5.6-terra | codex | BLOCKED | 12/20 | N/A/10 | N/A |
| gpt-5.6-luna | codex | BLOCKED | 0/20 | N/A/10 | N/A |
| gemini-3.8-flash-low | gemini_agy | COMPLETE | 20/20 | 5/10 | 0.5 |
| gemini-3.8-flash-high | gemini_agy | COMPLETE | 20/20 | 4/10 | 0.4 |
| gemini-3.8-flash-medium | gemini_agy | COMPLETE | 20/20 | 4/10 | 0.4 |
| claude-sonnet-4-6 | claude | COMPLETE | 20/20 | 8/10 | 0.8 |
| claude-opus-4-6-thinking | claude | COMPLETE | 20/20 | 6/10 | 0.6 |
