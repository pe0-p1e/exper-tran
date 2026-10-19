# Held-out-category generalization of the representation objective

User proposal, 2026-10-19: test categories beyond the original diverse-ten set
after fixing the loss, to establish whether the method generalizes across
categories. This is a planned study, not an executed result. It must not interrupt
the running Qwen/InternVL family-disjoint correction.

## Claim and mathematical scope

The class-prototype margin objective accepts C class scores; its formula is not
intrinsically restricted to C=10. However, the current manifests, reference-bank
loader, prompt and strict output parser are configured for the original ten
classes. Generalization needs its own versioned configuration and output root,
not a silent change to the frozen benchmark or the running runner.

The appropriate prospective claim is: reference-conditioned targeted transfer
generalizes to categories not used in method development. New class reference
images are still supplied, so this is not reference-free or zero-shot attack
construction. 'Unseen' means held out from experiment tuning, not proved absent
from a foundation model's pretraining data. Success here does not establish
general applicability to open-ended VQA, all semantic tasks or all models.

## Three separate questions

1. Category substitution: replace the category dictionary with a disjoint set of
   ten classes. Keep the objective/weights/layers/proxies/optimizer frozen.
2. Dictionary expansion: use nested10/20/50-class dictionaries, keeping the same
   attacked images and source/target pairs, changing only the distractor set.
3. Task transfer: separately test open-ended recognition/question answering.
   This remains deferred; never present closed-choice ASR as open-ended success.

Do not combine these three changes into one experiment and attribute any gain
or failure to the loss alone.

## Fixed prospective protocol

- Lock the method on the original open-weight surrogate development results
  before any new-category target outputs. Preserve strict BF16 proxy/target
  parameters for open-weight models, fixed epsilon16/255, 50 steps, seed42 and
 224px attack canvas. Maintain strict target-family exclusion.
- Select new classes from a versioned dataset/taxonomy with a recorded seed and
  explicit exclusion of all ten original classes. Record exact class/synset IDs
  before generation; do not choose successful classes after looking at results.
- Use48 references/class, disjoint from the new attack images. New references may
  instantiate the already-fixed objective; they may not tune its weights/layers.
- Distinguish old-source/new-target, new-source/old-target and new-source/new-target
  conditions if diagnosing where generalization fails. Freeze the pair list in
  advance rather than expanding into every combination immediately.
- Start with a very small predetermined interface/gradient smoke, then expand
  the same fixed pairs to a larger cohort. Report the pilot separately. Exact
  images/classes remain to be chosen after dataset availability is audited.
- For dictionary scaling, use fixed held-out source/target pairs contained in a
  nested10 subset of20 subset of50. The same pictures/reference IDs and all
  attack settings remain fixed; only extra competing class banks/options change.
- Report target targeted hits/N, clean accuracy, clean-correct-conditioned ASR,
  invalid outputs, per-direction confidence intervals, representation margins,
  DeltaR and compute. N=3 is a smoke, not evidence of broad generalization.
- Keep zero target-feedback algorithm selection. If target outcomes motivate a
  future method revision, start another declared development study and validate
  on independent held-out categories/images.

## Paper tables to preregister

TableG1: original10 versus disjoint new10, using the same frozen method, with
full-method and matched single controls and per-direction numerators/denominators.
Class sets differ here, so this is not a paired-image comparison across sets.

TableG2: nested10/20/50 options on identical new images/pairs, with clean accuracy,
targeted ASR and compute; paired-image comparisons are valid here.

TableG3, later: fixed-choice versus open-ended prompts on frozen images, with
predeclared semantic success criteria and independent response scoring. Do not
retune attacks from answers. This table is outside the current closed-choice run.
