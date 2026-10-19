# Independent historical result audit

Status: PASS_WITH_CAVEATS. Complete conditions: 25; PNGs: 7724; maximum pixel delta: 16/255; parser/hit mismatches: 0; failed historical dtype events: 0.

- Fixed ten-class forced-choice recognition, not open-ended QA.
- 30 images per direction; point estimates do not guarantee population ASR.
- Existing evaluator cache keys omit evaluator/prompt/parser hashes; no demonstrated inflation. New study locks include them.
- Historical cohort has been observed; further results are exploratory, requiring untouched confirmation.
- Fresh B1 versus B2 target replay not performed by this CPU audit.
