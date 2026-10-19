# Frozen-method commercial version comparison

Stage: test150; same fixed 150 images per model; clean/adversarial in fresh separate sessions.
Invalid predictions remain misses. Missing access/transport observations are N/A. All parameters/method/images are frozen before requests.
Model IDs are explicit requested client IDs; backend serving snapshots and closed weights are not exposed.
CLI routes and prompts are logged; different clients are not claimed to have identical system context.

| Model | Route | Status | Requests | Target hits / N | ASR |
|---|---|---|---|---|---|
| gpt-6.1-sol | codex | READY | 0/300 | N/A/150 | N/A |
| gpt-6-sol | codex | READY | 0/300 | N/A/150 | N/A |
| gpt-5.6-sol | codex | READY | 0/300 | N/A/150 | N/A |
| UNCONFIGURED | gemini | BLOCKED | 0/300 | N/A/150 | N/A |
| UNCONFIGURED | gemini | BLOCKED | 0/300 | N/A/150 | N/A |
| UNCONFIGURED | claude | BLOCKED | 0/300 | N/A/150 | N/A |
| UNCONFIGURED | claude | BLOCKED | 0/300 | N/A/150 | N/A |
