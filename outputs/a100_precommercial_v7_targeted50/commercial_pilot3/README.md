# Three-image commercial connectivity pilot

Prepared six separate image requests (three clean/adversarial pairs). No commercial
results have been measured. Files have opaque request IDs; target/source labels
are only in private/manifest.json and must never be supplied to a target client.

For every image use a fresh isolated session, the exact prompt.txt and only image.png.
Do not expose this repository, other requests, private manifests or attack status.
Record actual model ID, CLI version, route, UTC time, raw response and evidence that
one image was attached. Do not reuse conversation history or switch models on retry.

pilot3_blinded_inputs.zip contains only image.png/prompt.txt under opaque folders.
This is a limited three-direction interface smoke, not a completed scientific
ablation or a statistical transferability claim. The GPU campaign runs separately.
