# Complete final150 image pairs

Each ZIP contains150 shared clean PNGs and450 adversarial PNGs (Uniform, Selected and FDA,150 each), plus exact settings, image IDs, per-file SHA256 and pairing manifest. The clean images are shared rather than copied three times. All frozen PNG bytes are preserved.

- `final150_qwen27b_and_all_gpt_clean_adv.zip`: Qwen27B and all evaluated GPT models;60,433,975bytes.
- `final150_internvl14b_clean_adv.zip`: InternVL14B;60,482,910bytes.
- `final150_gemma31b_clean_adv.zip`: Gemma31B;60,371,322bytes.

The three target-family-excluded proxy pools generate different adversarial PNGs. Each ZIP preserves all three frozen methods. Across ZIPs there are1350 adversarial images and150 distinct clean images; clean images are repeated across ZIPs for independent use. See index.json for archive SHA256. Remaining GPT5.6 evaluations were cancelled by the user; their input images remain complete, and incomplete outcomes are not called final150 evaluations.
