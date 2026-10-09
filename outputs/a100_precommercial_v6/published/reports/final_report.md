# A100 Pre-Commercial V6 Experiment Report

Generated UTC: 2026-10-09T12:59:21.139845+00:00
Summed measured attack GPU runtime (cell-level): 14.64 hours (52705 seconds). This sum excludes model audit, clean screening, calibration, and representation extraction.

## Protocol

Open-source models only. Pull/Push = 0.75/0.25 for the main method, CLS=0, epsilon=16/255, 50 steps, step size=1/255, momentum=1, random start, seed=42. TASR is target targeted success conditioned on proxy targeted success; numerator and denominator are exported explicitly. Multiple-proxy primary conditioning requires all ensemble proxies to hit.

## Model revisions

| Model | Repository | Revision | Vision depth |
|---|---|---|---:|
| Qwen3.5-2B | Qwen/Qwen3.5-2B | `15852e8c16360a2fea060d615a32b45270f8a8fc` | 24 |
| Qwen3.5-4B | Qwen/Qwen3.5-4B | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | 24 |
| Qwen3.5-9B | Qwen/Qwen3.5-9B | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | 27 |
| Qwen3.5-27B | Qwen/Qwen3.5-27B | `fc05daec18b0a78c049392ed2e771dde82bdf654` | 27 |
| InternVL3.5-2B-HF | OpenGVLab/InternVL3_5-2B-HF | `3f301ffcf3dcbb47893afae6650ea3e78d96fb6d` | 24 |
| InternVL3.5-4B-HF | OpenGVLab/InternVL3_5-4B-HF | `6bd4487402110ef9889ba50eb7aefeb302526fed` | 24 |
| InternVL3.5-8B-HF | OpenGVLab/InternVL3_5-8B-HF | `741a7d03020411e666c6109218ab71e08151ef86` | 24 |
| InternVL3.5-14B-HF | OpenGVLab/InternVL3_5-14B-HF | `226b96d5912e69159abc0384cefcbd51487fdce0` | 24 |
| Gemma 4 E2B-it | google/gemma-4-E2B-it | `3e22461f65e89153144f8adb70e3b8c2cc9845a7` | 16 |
| Gemma 4 E4B-it | google/gemma-4-E4B-it | `ee0ef6023621cff504d758262d4e04895a5af4a2` | 16 |
| Gemma 4 26B-A4B-it | google/gemma-4-26B-A4B-it | `4d7ae4984b7db7de8f8457170b3f1a419ee76d52` | 27 |
| Gemma 4 31B-it | google/gemma-4-31B-it | `842da3794eaa0b77d5f08bae87a17459d91ff475` | 27 |
| CLIP ViT-L/14 | openai/clip-vit-large-patch14 | `32bd64288804d66eefd0ccbe215aa642df71cc41` | 24 |
| SigLIP2-So400m-patch14-384 | google/siglip2-so400m-patch14-384 | `e8e487298228002f3d8a82e0cd5c8ea9c567f57f` | 27 |
| DINOv2-Large | facebook/dinov2-large | `47b73eefe95e8d44ec3623f8890bd894b6ea2d6c` | 24 |

## Completed measurements

Measured transition cells: 368; per-image rows: 11040; class dispersion rows: 90.

Completion-matrix coverage, including screened ensemble shortfalls:
- ablation: complete = 50
- layer_sweep: complete = 150
- multiple_proxy: complete = 48
- multiple_proxy: insufficient_clean_valid = 17
- reverse_direction: complete = 60
- single_proxy: complete = 60

| Family | Proxy | Target | Direction | Layer | TASR numerator / denominator | TASR | ΔR | CKA | RSA |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| single_proxy | InternVL3.5-2B-HF | Gemma 4 E4B-it | goldfish → monarch butterfly | 23 | 1/30 | 0.033 | 0.0039 | 0.9368488788604736 | 0.8553114802759518 |
| single_proxy | InternVL3.5-2B-HF | Gemma 4 E4B-it | pineapple → acoustic guitar | 23 | 0/30 | 0.000 | 0.0163 | 0.8564447164535522 | 0.6547870531027303 |
| single_proxy | InternVL3.5-2B-HF | Gemma 4 E4B-it | laptop → espresso | 23 | 0/30 | 0.000 | 0.0031 | 0.9002621173858643 | 0.7505661835450219 |
| single_proxy | InternVL3.5-2B-HF | Gemma 4 E4B-it | volcano → rocking chair | 23 | 1/30 | 0.033 | 0.0098 | 0.9298343658447266 | 0.7992798860584072 |
| single_proxy | InternVL3.5-2B-HF | Gemma 4 E4B-it | soccer ball → school bus | 23 | 2/30 | 0.067 | 0.0019 | 0.8798832297325134 | 0.728837737748348 |
| single_proxy | InternVL3.5-2B-HF | InternVL3.5-4B-HF | goldfish → monarch butterfly | 23 | 25/30 | 0.833 | 0.0312 | 0.9660873413085938 | 0.9137480511021122 |
| single_proxy | InternVL3.5-2B-HF | InternVL3.5-4B-HF | pineapple → acoustic guitar | 23 | 6/30 | 0.200 | 0.0218 | 0.9570565819740295 | 0.9058564879589585 |
| single_proxy | InternVL3.5-2B-HF | InternVL3.5-4B-HF | laptop → espresso | 23 | 4/29 | 0.138 | 0.0066 | 0.9477026462554932 | 0.8925706957395585 |
| single_proxy | InternVL3.5-2B-HF | InternVL3.5-4B-HF | volcano → rocking chair | 23 | 18/30 | 0.600 | 0.0458 | 0.974137544631958 | 0.9261488980194571 |
| single_proxy | InternVL3.5-2B-HF | InternVL3.5-4B-HF | soccer ball → school bus | 23 | 0/30 | 0.000 | 0.0086 | 0.9472202062606812 | 0.8764245652236676 |
| single_proxy | Gemma 4 E2B-it | Gemma 4 26B-A4B-it | goldfish → monarch butterfly | 15 | 2/30 | 0.067 | 0.0070 | 0.9571262001991272 | 0.9026865443320575 |
| single_proxy | Gemma 4 E2B-it | Gemma 4 26B-A4B-it | pineapple → acoustic guitar | 15 | 0/30 | 0.000 | 0.0099 | 0.9322985410690308 | 0.8886508062939197 |
| single_proxy | Gemma 4 E2B-it | Gemma 4 26B-A4B-it | laptop → espresso | 15 | 1/30 | 0.033 | 0.0067 | 0.9033745527267456 | 0.8599638076943454 |
| single_proxy | Gemma 4 E2B-it | Gemma 4 26B-A4B-it | volcano → rocking chair | 15 | 3/30 | 0.100 | 0.0081 | 0.9484579563140869 | 0.888266571553616 |
| single_proxy | Gemma 4 E2B-it | Gemma 4 26B-A4B-it | soccer ball → school bus | 15 | 0/30 | 0.000 | 0.0030 | 0.9266352653503418 | 0.8937779149251907 |
| single_proxy | Gemma 4 E2B-it | Gemma 4 E4B-it | goldfish → monarch butterfly | 15 | 30/30 | 1.000 | 0.0521 | 1.0 | 1.0 |
| single_proxy | Gemma 4 E2B-it | Gemma 4 E4B-it | pineapple → acoustic guitar | 15 | 30/30 | 1.000 | 0.0815 | 1.0 | 1.0 |
| single_proxy | Gemma 4 E2B-it | Gemma 4 E4B-it | laptop → espresso | 15 | 30/30 | 1.000 | 0.0307 | 1.0 | 0.9999999999999998 |
| single_proxy | Gemma 4 E2B-it | Gemma 4 E4B-it | volcano → rocking chair | 15 | 30/30 | 1.000 | 0.0579 | 1.0 | 0.9999999999999998 |
| single_proxy | Gemma 4 E2B-it | Gemma 4 E4B-it | soccer ball → school bus | 15 | 30/30 | 1.000 | 0.0250 | 1.0 | 1.0 |
| single_proxy | Qwen3.5-2B | Gemma 4 E4B-it | goldfish → monarch butterfly | 23 | 4/30 | 0.133 | 0.0050 | 0.8509443402290344 | 0.7120382811920831 |
| single_proxy | Qwen3.5-2B | Gemma 4 E4B-it | pineapple → acoustic guitar | 23 | 0/29 | 0.000 | 0.0149 | 0.7642777562141418 | 0.5277742813395394 |
| single_proxy | Qwen3.5-2B | Gemma 4 E4B-it | laptop → espresso | 23 | 0/12 | 0.000 | 0.0022 | 0.7853065133094788 | 0.6010451529782691 |
| single_proxy | Qwen3.5-2B | Gemma 4 E4B-it | volcano → rocking chair | 23 | 1/30 | 0.033 | 0.0118 | 0.8427858948707581 | 0.6248395483169762 |
| single_proxy | Qwen3.5-2B | Gemma 4 E4B-it | soccer ball → school bus | 23 | 0/24 | 0.000 | 0.0023 | 0.8269062638282776 | 0.6815021936859391 |
| single_proxy | Gemma 4 E2B-it | InternVL3.5-4B-HF | goldfish → monarch butterfly | 15 | 7/30 | 0.233 | 0.0144 | 0.9418367147445679 | 0.8409391763459584 |
| single_proxy | Gemma 4 E2B-it | InternVL3.5-4B-HF | pineapple → acoustic guitar | 15 | 0/30 | 0.000 | 0.0126 | 0.9085942506790161 | 0.750940208052508 |
| single_proxy | Gemma 4 E2B-it | InternVL3.5-4B-HF | laptop → espresso | 15 | 0/30 | 0.000 | 0.0037 | 0.902607798576355 | 0.738224495456555 |
| single_proxy | Gemma 4 E2B-it | InternVL3.5-4B-HF | volcano → rocking chair | 15 | 7/30 | 0.233 | 0.0341 | 0.928453803062439 | 0.7981106763834439 |
| single_proxy | Gemma 4 E2B-it | InternVL3.5-4B-HF | soccer ball → school bus | 15 | 1/30 | 0.033 | 0.0062 | 0.8687722086906433 | 0.7181038222246017 |
| single_proxy | Qwen3.5-2B | Qwen3.5-9B | goldfish → monarch butterfly | 23 | 10/30 | 0.333 | 0.0000 | 0.9408533573150635 | 0.8766009203812413 |
| single_proxy | Qwen3.5-2B | Qwen3.5-9B | pineapple → acoustic guitar | 23 | 0/29 | 0.000 | 0.0000 | 0.9145583510398865 | 0.8489998390409518 |
| single_proxy | Qwen3.5-2B | Qwen3.5-9B | laptop → espresso | 23 | 0/12 | 0.000 | 0.0000 | 0.837624192237854 | 0.6445863072178778 |
| single_proxy | Qwen3.5-2B | Qwen3.5-9B | volcano → rocking chair | 23 | 9/30 | 0.300 | 0.0000 | 0.9071047902107239 | 0.7958412166731171 |
| single_proxy | Qwen3.5-2B | Qwen3.5-9B | soccer ball → school bus | 23 | 1/24 | 0.042 | 0.0000 | 0.8854109048843384 | 0.7975380583077056 |
| single_proxy | InternVL3.5-2B-HF | Qwen3.5-4B | goldfish → monarch butterfly | 23 | 1/30 | 0.033 | 0.0000 | 0.9007353186607361 | 0.8068765031340197 |
| single_proxy | InternVL3.5-2B-HF | Qwen3.5-4B | pineapple → acoustic guitar | 23 | 0/30 | 0.000 | 0.0000 | 0.9254664182662964 | 0.8527184915490558 |
| single_proxy | InternVL3.5-2B-HF | Qwen3.5-4B | laptop → espresso | 23 | 0/30 | 0.000 | 0.0000 | 0.8917120695114136 | 0.7328102200688649 |
| single_proxy | InternVL3.5-2B-HF | Qwen3.5-4B | volcano → rocking chair | 23 | 0/30 | 0.000 | 0.0001 | 0.8988475203514099 | 0.7617233282285629 |
| single_proxy | InternVL3.5-2B-HF | Qwen3.5-4B | soccer ball → school bus | 23 | 0/30 | 0.000 | 0.0000 | 0.8831189870834351 | 0.7411675301760164 |
| single_proxy | Qwen3.5-2B | Qwen3.5-4B | goldfish → monarch butterfly | 23 | 29/30 | 0.967 | 0.0002 | 0.9572506546974182 | 0.919183994272392 |
| single_proxy | Qwen3.5-2B | Qwen3.5-4B | pineapple → acoustic guitar | 23 | 17/29 | 0.586 | 0.0001 | 0.9611946940422058 | 0.9094387122705928 |
| single_proxy | Qwen3.5-2B | Qwen3.5-4B | laptop → espresso | 23 | 6/12 | 0.500 | 0.0000 | 0.9238079786300659 | 0.8302092751133474 |
| single_proxy | Qwen3.5-2B | Qwen3.5-4B | volcano → rocking chair | 23 | 30/30 | 1.000 | 0.0002 | 0.9635360836982727 | 0.895682610392392 |
| single_proxy | Qwen3.5-2B | Qwen3.5-4B | soccer ball → school bus | 23 | 10/24 | 0.417 | 0.0000 | 0.9427202939987183 | 0.8640513607699123 |
| single_proxy | InternVL3.5-2B-HF | InternVL3.5-8B-HF | goldfish → monarch butterfly | 23 | 18/30 | 0.600 | 0.0288 | 0.9679558873176575 | 0.9327099421440115 |
| single_proxy | InternVL3.5-2B-HF | InternVL3.5-8B-HF | pineapple → acoustic guitar | 23 | 1/30 | 0.033 | 0.0177 | 0.9790648818016052 | 0.9532871806750695 |
| single_proxy | InternVL3.5-2B-HF | InternVL3.5-8B-HF | laptop → espresso | 23 | 0/29 | 0.000 | 0.0070 | 0.9495866894721985 | 0.9061634809251022 |
| single_proxy | InternVL3.5-2B-HF | InternVL3.5-8B-HF | volcano → rocking chair | 23 | 12/30 | 0.400 | 0.0394 | 0.9691879749298096 | 0.912080013164289 |
| single_proxy | InternVL3.5-2B-HF | InternVL3.5-8B-HF | soccer ball → school bus | 23 | 0/30 | 0.000 | 0.0104 | 0.9484160542488098 | 0.8717490832323199 |
| single_proxy | Qwen3.5-2B | InternVL3.5-4B-HF | goldfish → monarch butterfly | 23 | 10/30 | 0.333 | 0.0145 | 0.8760660290718079 | 0.7532517662009103 |
| single_proxy | Qwen3.5-2B | InternVL3.5-4B-HF | pineapple → acoustic guitar | 23 | 0/29 | 0.000 | 0.0070 | 0.8807937502861023 | 0.7965858369976894 |
| single_proxy | Qwen3.5-2B | InternVL3.5-4B-HF | laptop → espresso | 23 | 0/12 | 0.000 | 0.0018 | 0.8259283304214478 | 0.6303236516918393 |
| single_proxy | Qwen3.5-2B | InternVL3.5-4B-HF | volcano → rocking chair | 23 | 3/30 | 0.100 | 0.0282 | 0.8639575242996216 | 0.7222413740967455 |
| single_proxy | Qwen3.5-2B | InternVL3.5-4B-HF | soccer ball → school bus | 23 | 0/25 | 0.000 | 0.0038 | 0.8293028473854065 | 0.7120069379126106 |
| single_proxy | Gemma 4 E2B-it | Qwen3.5-4B | goldfish → monarch butterfly | 15 | 0/30 | 0.000 | 0.0001 | 0.8674367666244507 | 0.7460722647620969 |
| single_proxy | Gemma 4 E2B-it | Qwen3.5-4B | pineapple → acoustic guitar | 15 | 0/30 | 0.000 | 0.0000 | 0.7956522703170776 | 0.5024936072839901 |
| single_proxy | Gemma 4 E2B-it | Qwen3.5-4B | laptop → espresso | 15 | 0/30 | 0.000 | 0.0000 | 0.8433841466903687 | 0.6087014194376508 |
| single_proxy | Gemma 4 E2B-it | Qwen3.5-4B | volcano → rocking chair | 15 | 4/30 | 0.133 | 0.0001 | 0.8667718172073364 | 0.6194269359076255 |
| single_proxy | Gemma 4 E2B-it | Qwen3.5-4B | soccer ball → school bus | 15 | 0/30 | 0.000 | 0.0000 | 0.8193196058273315 | 0.5438602368238711 |
| layer_sweep | Gemma 4 E2B-it | Gemma 4 E4B-it | goldfish → monarch butterfly | 0 | 0/0 | — | 0.0008 | 0.1736457347869873 | 0.23490790070349415 |
| layer_sweep | Gemma 4 E2B-it | Gemma 4 E4B-it | pineapple → acoustic guitar | 0 | 0/0 | — | 0.0063 | 0.16747531294822693 | 0.08457524118631007 |
| layer_sweep | Gemma 4 E2B-it | Gemma 4 E4B-it | laptop → espresso | 0 | 0/0 | — | 0.0009 | 0.19166764616966248 | 0.28512577598361083 |
| layer_sweep | Gemma 4 E2B-it | Gemma 4 E4B-it | volcano → rocking chair | 0 | 0/0 | — | 0.0042 | 0.2085106372833252 | 0.12530982632109008 |
| layer_sweep | Gemma 4 E2B-it | Gemma 4 E4B-it | soccer ball → school bus | 0 | 0/0 | — | 0.0005 | 0.21694041788578033 | 0.3358645377798367 |
| layer_sweep | Qwen3.5-2B | Gemma 4 E4B-it | goldfish → monarch butterfly | 23 | 4/30 | 0.133 | 0.0050 | 0.8509443402290344 | 0.7120382811920831 |
| layer_sweep | Qwen3.5-2B | Gemma 4 E4B-it | pineapple → acoustic guitar | 23 | 0/29 | 0.000 | 0.0148 | 0.7642777562141418 | 0.5277742813395394 |
| layer_sweep | Qwen3.5-2B | Gemma 4 E4B-it | laptop → espresso | 23 | 0/12 | 0.000 | 0.0022 | 0.7853065133094788 | 0.6010451529782691 |
| layer_sweep | Qwen3.5-2B | Gemma 4 E4B-it | volcano → rocking chair | 23 | 1/30 | 0.033 | 0.0119 | 0.8427858948707581 | 0.6248395483169762 |
| layer_sweep | Qwen3.5-2B | Gemma 4 E4B-it | soccer ball → school bus | 23 | 0/24 | 0.000 | 0.0023 | 0.8269062638282776 | 0.6815021936859391 |
| layer_sweep | InternVL3.5-2B-HF | Qwen3.5-4B | goldfish → monarch butterfly | 17 | 0/30 | 0.000 | 0.0001 | 0.8708397746086121 | 0.7525089835565927 |
| layer_sweep | InternVL3.5-2B-HF | Qwen3.5-4B | pineapple → acoustic guitar | 17 | 0/30 | 0.000 | 0.0000 | 0.8972452282905579 | 0.780023311470916 |
| layer_sweep | InternVL3.5-2B-HF | Qwen3.5-4B | laptop → espresso | 17 | 0/29 | 0.000 | 0.0000 | 0.8977063894271851 | 0.7393359100699366 |
| layer_sweep | InternVL3.5-2B-HF | Qwen3.5-4B | volcano → rocking chair | 17 | 2/30 | 0.067 | 0.0001 | 0.9088118672370911 | 0.7700049210094776 |
| layer_sweep | InternVL3.5-2B-HF | Qwen3.5-4B | soccer ball → school bus | 17 | 0/30 | 0.000 | 0.0000 | 0.8883188366889954 | 0.6866266292785389 |
| layer_sweep | Gemma 4 E2B-it | InternVL3.5-8B-HF | goldfish → monarch butterfly | 11 | 1/17 | 0.059 | 0.0140 | 0.6911462545394897 | 0.6647097078868411 |
| layer_sweep | Gemma 4 E2B-it | InternVL3.5-8B-HF | pineapple → acoustic guitar | 11 | 0/11 | 0.000 | 0.0105 | 0.8262580633163452 | 0.7365403698937181 |
| layer_sweep | Gemma 4 E2B-it | InternVL3.5-8B-HF | laptop → espresso | 11 | 0/22 | 0.000 | 0.0068 | 0.7455505728721619 | 0.6531770189576944 |
| layer_sweep | Gemma 4 E2B-it | InternVL3.5-8B-HF | volcano → rocking chair | 11 | 13/30 | 0.433 | 0.0471 | 0.7811850905418396 | 0.6566092176500848 |
| layer_sweep | Gemma 4 E2B-it | InternVL3.5-8B-HF | soccer ball → school bus | 11 | 3/25 | 0.120 | 0.0132 | 0.7476949691772461 | 0.730546859484267 |
| layer_sweep | Qwen3.5-2B | Qwen3.5-4B | goldfish → monarch butterfly | 23 | 29/30 | 0.967 | 0.0002 | 0.9572506546974182 | 0.919183994272392 |
| layer_sweep | Qwen3.5-2B | Qwen3.5-4B | pineapple → acoustic guitar | 23 | 16/29 | 0.552 | 0.0001 | 0.9611946940422058 | 0.9094387122705928 |
| layer_sweep | Qwen3.5-2B | Qwen3.5-4B | laptop → espresso | 23 | 6/12 | 0.500 | 0.0000 | 0.9238079786300659 | 0.8302092751133474 |
| layer_sweep | Qwen3.5-2B | Qwen3.5-4B | volcano → rocking chair | 23 | 30/30 | 1.000 | 0.0002 | 0.9635360836982727 | 0.895682610392392 |
| layer_sweep | Qwen3.5-2B | Qwen3.5-4B | soccer ball → school bus | 23 | 10/24 | 0.417 | 0.0000 | 0.9427202939987183 | 0.8640513607699123 |
| layer_sweep | Gemma 4 E2B-it | Gemma 4 E4B-it | goldfish → monarch butterfly | 7 | 0/0 | — | 0.0063 | 0.5085483193397522 | 0.43403910867514833 |
| layer_sweep | Gemma 4 E2B-it | Gemma 4 E4B-it | pineapple → acoustic guitar | 7 | 0/0 | — | 0.0241 | 0.6154216527938843 | 0.376534978069615 |
| layer_sweep | Gemma 4 E2B-it | Gemma 4 E4B-it | laptop → espresso | 7 | 0/1 | 0.000 | 0.0062 | 0.690237820148468 | 0.5957175549052273 |
| layer_sweep | Gemma 4 E2B-it | Gemma 4 E4B-it | volcano → rocking chair | 7 | 0/0 | — | 0.0156 | 0.48777106404304504 | 0.3227897708785175 |
| layer_sweep | Gemma 4 E2B-it | Gemma 4 E4B-it | soccer ball → school bus | 7 | 0/0 | — | 0.0055 | 0.6120365858078003 | 0.6702365041056244 |
| layer_sweep | Qwen3.5-2B | Qwen3.5-4B | goldfish → monarch butterfly | 0 | 0/0 | — | 0.0000 | 0.30759283900260925 | 0.44019535077614486 |
| layer_sweep | Qwen3.5-2B | Qwen3.5-4B | pineapple → acoustic guitar | 0 | 0/0 | — | 0.0000 | 0.4414984881877899 | 0.4933468776587799 |
| layer_sweep | Qwen3.5-2B | Qwen3.5-4B | laptop → espresso | 0 | 0/0 | — | 0.0000 | 0.3595409691333771 | 0.41435384958870375 |
| layer_sweep | Qwen3.5-2B | Qwen3.5-4B | volcano → rocking chair | 0 | 0/0 | — | 0.0000 | 0.3886905610561371 | 0.30490284944227247 |
| layer_sweep | Qwen3.5-2B | Qwen3.5-4B | soccer ball → school bus | 0 | 0/0 | — | -0.0000 | 0.36920464038848877 | 0.34848965959185296 |
| layer_sweep | InternVL3.5-2B-HF | InternVL3.5-4B-HF | goldfish → monarch butterfly | 23 | 26/30 | 0.867 | 0.0323 | 0.9660873413085938 | 0.9137480511021122 |
| layer_sweep | InternVL3.5-2B-HF | InternVL3.5-4B-HF | pineapple → acoustic guitar | 23 | 7/30 | 0.233 | 0.0212 | 0.9570565819740295 | 0.9058564879589585 |
| layer_sweep | InternVL3.5-2B-HF | InternVL3.5-4B-HF | laptop → espresso | 23 | 5/30 | 0.167 | 0.0066 | 0.9477026462554932 | 0.8925706957395585 |
| layer_sweep | InternVL3.5-2B-HF | InternVL3.5-4B-HF | volcano → rocking chair | 23 | 19/30 | 0.633 | 0.0461 | 0.974137544631958 | 0.9261488980194571 |
| layer_sweep | InternVL3.5-2B-HF | InternVL3.5-4B-HF | soccer ball → school bus | 23 | 1/30 | 0.033 | 0.0087 | 0.9472202062606812 | 0.8764245652236676 |
| layer_sweep | Gemma 4 E2B-it | InternVL3.5-8B-HF | goldfish → monarch butterfly | 15 | 2/30 | 0.067 | 0.0146 | 0.9446554183959961 | 0.8513684194599561 |
| layer_sweep | Gemma 4 E2B-it | InternVL3.5-8B-HF | pineapple → acoustic guitar | 15 | 1/30 | 0.033 | 0.0091 | 0.8773710131645203 | 0.6844952962272111 |
| layer_sweep | Gemma 4 E2B-it | InternVL3.5-8B-HF | laptop → espresso | 15 | 0/30 | 0.000 | 0.0036 | 0.9144735336303711 | 0.7870475538548771 |
| layer_sweep | Gemma 4 E2B-it | InternVL3.5-8B-HF | volcano → rocking chair | 15 | 4/30 | 0.133 | 0.0303 | 0.928925633430481 | 0.7911528179319045 |
| layer_sweep | Gemma 4 E2B-it | InternVL3.5-8B-HF | soccer ball → school bus | 15 | 0/30 | 0.000 | 0.0084 | 0.8964642882347107 | 0.7883085049067191 |
| layer_sweep | Qwen3.5-2B | Gemma 4 E4B-it | goldfish → monarch butterfly | 11 | 2/29 | 0.069 | 0.0041 | 0.24710965156555176 | 0.3079925925412732 |
| layer_sweep | Qwen3.5-2B | Gemma 4 E4B-it | pineapple → acoustic guitar | 11 | 0/29 | 0.000 | 0.0223 | 0.29029834270477295 | 0.2435799123650688 |
| layer_sweep | Qwen3.5-2B | Gemma 4 E4B-it | laptop → espresso | 11 | 0/9 | 0.000 | 0.0046 | 0.357486754655838 | 0.40253643089293084 |
| layer_sweep | Qwen3.5-2B | Gemma 4 E4B-it | volcano → rocking chair | 11 | 2/28 | 0.071 | 0.0161 | 0.2884472906589508 | 0.22687115030866156 |
| layer_sweep | Qwen3.5-2B | Gemma 4 E4B-it | soccer ball → school bus | 11 | 2/17 | 0.118 | 0.0040 | 0.32222461700439453 | 0.37100460576688915 |
| layer_sweep | InternVL3.5-2B-HF | Qwen3.5-4B | goldfish → monarch butterfly | 23 | 1/30 | 0.033 | 0.0000 | 0.9007353186607361 | 0.8068765031340197 |
| layer_sweep | InternVL3.5-2B-HF | Qwen3.5-4B | pineapple → acoustic guitar | 23 | 0/30 | 0.000 | 0.0000 | 0.9254664182662964 | 0.8527184915490558 |
| layer_sweep | InternVL3.5-2B-HF | Qwen3.5-4B | laptop → espresso | 23 | 0/29 | 0.000 | 0.0000 | 0.8917120695114136 | 0.7328102200688649 |
| layer_sweep | InternVL3.5-2B-HF | Qwen3.5-4B | volcano → rocking chair | 23 | 0/30 | 0.000 | 0.0001 | 0.8988475203514099 | 0.7617233282285629 |
| layer_sweep | InternVL3.5-2B-HF | Qwen3.5-4B | soccer ball → school bus | 23 | 0/30 | 0.000 | 0.0000 | 0.8831189870834351 | 0.7411675301760164 |
| layer_sweep | Qwen3.5-2B | Qwen3.5-4B | goldfish → monarch butterfly | 17 | 30/30 | 1.000 | 0.0002 | 0.8650460839271545 | 0.8157579612434802 |
| layer_sweep | Qwen3.5-2B | Qwen3.5-4B | pineapple → acoustic guitar | 17 | 30/30 | 1.000 | 0.0001 | 0.8333829641342163 | 0.8142876536918714 |
| layer_sweep | Qwen3.5-2B | Qwen3.5-4B | laptop → espresso | 17 | 14/28 | 0.500 | 0.0000 | 0.7669875025749207 | 0.6946947812593313 |
| layer_sweep | Qwen3.5-2B | Qwen3.5-4B | volcano → rocking chair | 17 | 30/30 | 1.000 | 0.0002 | 0.7715232372283936 | 0.6595009076835613 |
| layer_sweep | Qwen3.5-2B | Qwen3.5-4B | soccer ball → school bus | 17 | 22/30 | 0.733 | 0.0000 | 0.8032207489013672 | 0.6531132728699035 |
| layer_sweep | Qwen3.5-2B | Gemma 4 E4B-it | goldfish → monarch butterfly | 0 | 0/0 | — | 0.0009 | 0.308555006980896 | 0.3530339272839954 |
| layer_sweep | Qwen3.5-2B | Gemma 4 E4B-it | pineapple → acoustic guitar | 0 | 0/0 | — | 0.0118 | 0.22449085116386414 | 0.15086118377601604 |
| layer_sweep | Qwen3.5-2B | Gemma 4 E4B-it | laptop → espresso | 0 | 0/0 | — | 0.0021 | 0.2705369293689728 | 0.37643814534202463 |
| layer_sweep | Qwen3.5-2B | Gemma 4 E4B-it | volcano → rocking chair | 0 | 0/0 | — | 0.0055 | 0.2573111355304718 | 0.19121418479988678 |
| layer_sweep | Qwen3.5-2B | Gemma 4 E4B-it | soccer ball → school bus | 0 | 0/0 | — | 0.0009 | 0.3188392221927643 | 0.4455475706820317 |
| layer_sweep | Qwen3.5-2B | Gemma 4 E4B-it | goldfish → monarch butterfly | 17 | 3/30 | 0.100 | 0.0051 | 0.8654278516769409 | 0.7886663939301112 |
| layer_sweep | Qwen3.5-2B | Gemma 4 E4B-it | pineapple → acoustic guitar | 17 | 0/30 | 0.000 | 0.0172 | 0.6781435012817383 | 0.5064125532834739 |
| layer_sweep | Qwen3.5-2B | Gemma 4 E4B-it | laptop → espresso | 17 | 0/28 | 0.000 | 0.0031 | 0.676613450050354 | 0.6291877718513447 |
| layer_sweep | Qwen3.5-2B | Gemma 4 E4B-it | volcano → rocking chair | 17 | 1/30 | 0.033 | 0.0121 | 0.6859338879585266 | 0.5553113525611243 |
| layer_sweep | Qwen3.5-2B | Gemma 4 E4B-it | soccer ball → school bus | 17 | 2/30 | 0.067 | 0.0031 | 0.7315370440483093 | 0.7076131878158188 |
| layer_sweep | Gemma 4 E2B-it | Gemma 4 E4B-it | goldfish → monarch butterfly | 11 | 16/17 | 0.941 | 0.0245 | 0.7112680673599243 | 0.6431805865130277 |
| layer_sweep | Gemma 4 E2B-it | Gemma 4 E4B-it | pineapple → acoustic guitar | 11 | 13/13 | 1.000 | 0.0503 | 0.8212580680847168 | 0.6737322544336147 |
| layer_sweep | Gemma 4 E2B-it | Gemma 4 E4B-it | laptop → espresso | 11 | 18/20 | 0.900 | 0.0184 | 0.8389164805412292 | 0.742322862857516 |
| layer_sweep | Gemma 4 E2B-it | Gemma 4 E4B-it | volcano → rocking chair | 11 | 28/29 | 0.966 | 0.0439 | 0.8410777449607849 | 0.6937714748914161 |
| layer_sweep | Gemma 4 E2B-it | Gemma 4 E4B-it | soccer ball → school bus | 11 | 24/25 | 0.960 | 0.0169 | 0.8408633470535278 | 0.8447427703650441 |
| layer_sweep | Gemma 4 E2B-it | InternVL3.5-8B-HF | goldfish → monarch butterfly | 3 | 0/0 | — | 0.0052 | 0.24694758653640747 | 0.4058994049005963 |
| layer_sweep | Gemma 4 E2B-it | InternVL3.5-8B-HF | pineapple → acoustic guitar | 3 | 0/0 | — | -0.0001 | 0.39380764961242676 | 0.48135645332933147 |
| layer_sweep | Gemma 4 E2B-it | InternVL3.5-8B-HF | laptop → espresso | 3 | 0/0 | — | 0.0000 | 0.25491926074028015 | 0.30423615251044933 |
| layer_sweep | Gemma 4 E2B-it | InternVL3.5-8B-HF | volcano → rocking chair | 3 | 0/0 | — | 0.0182 | 0.24813280999660492 | 0.21081533372207645 |
| layer_sweep | Gemma 4 E2B-it | InternVL3.5-8B-HF | soccer ball → school bus | 3 | 0/0 | — | 0.0034 | 0.308550626039505 | 0.4513559121998392 |
| layer_sweep | Gemma 4 E2B-it | InternVL3.5-8B-HF | goldfish → monarch butterfly | 7 | 0/0 | — | 0.0071 | 0.5027905702590942 | 0.5123357210125403 |
| layer_sweep | Gemma 4 E2B-it | InternVL3.5-8B-HF | pineapple → acoustic guitar | 7 | 0/0 | — | 0.0033 | 0.6745110154151917 | 0.609963437960143 |
| layer_sweep | Gemma 4 E2B-it | InternVL3.5-8B-HF | laptop → espresso | 7 | 0/0 | — | 0.0023 | 0.619692862033844 | 0.51655618906926 |
| layer_sweep | Gemma 4 E2B-it | InternVL3.5-8B-HF | volcano → rocking chair | 7 | 0/0 | — | 0.0231 | 0.46339765191078186 | 0.3618261920078254 |
| layer_sweep | Gemma 4 E2B-it | InternVL3.5-8B-HF | soccer ball → school bus | 7 | 0/0 | — | 0.0074 | 0.5765939354896545 | 0.6089505222686683 |
| layer_sweep | Qwen3.5-2B | Qwen3.5-4B | goldfish → monarch butterfly | 5 | 0/0 | — | 0.0001 | 0.24238266050815582 | 0.3286270579767144 |
| layer_sweep | Qwen3.5-2B | Qwen3.5-4B | pineapple → acoustic guitar | 5 | 0/0 | — | 0.0000 | 0.47407376766204834 | 0.4751177367804922 |
| layer_sweep | Qwen3.5-2B | Qwen3.5-4B | laptop → espresso | 5 | 0/0 | — | 0.0000 | 0.3779006898403168 | 0.4140935397208159 |
| layer_sweep | Qwen3.5-2B | Qwen3.5-4B | volcano → rocking chair | 5 | 0/0 | — | 0.0001 | 0.37952643632888794 | 0.3142938831285422 |
| layer_sweep | Qwen3.5-2B | Qwen3.5-4B | soccer ball → school bus | 5 | 0/0 | — | 0.0000 | 0.3778315782546997 | 0.34954815533741795 |
| layer_sweep | Gemma 4 E2B-it | InternVL3.5-8B-HF | goldfish → monarch butterfly | 0 | 0/0 | — | 0.0044 | 0.151076078414917 | 0.31139527596549044 |
| layer_sweep | Gemma 4 E2B-it | InternVL3.5-8B-HF | pineapple → acoustic guitar | 0 | 0/0 | — | -0.0004 | 0.25682497024536133 | 0.3682055555265005 |
| layer_sweep | Gemma 4 E2B-it | InternVL3.5-8B-HF | laptop → espresso | 0 | 0/0 | — | -0.0000 | 0.17740754783153534 | 0.24860152091342297 |
| layer_sweep | Gemma 4 E2B-it | InternVL3.5-8B-HF | volcano → rocking chair | 0 | 0/0 | — | 0.0118 | 0.2078651636838913 | 0.17888269255466097 |
| layer_sweep | Gemma 4 E2B-it | InternVL3.5-8B-HF | soccer ball → school bus | 0 | 0/0 | — | 0.0028 | 0.18891645967960358 | 0.293413334942792 |
| layer_sweep | Gemma 4 E2B-it | Gemma 4 E4B-it | goldfish → monarch butterfly | 3 | 0/0 | — | 0.0022 | 0.26484930515289307 | 0.32254924919035804 |
| layer_sweep | Gemma 4 E2B-it | Gemma 4 E4B-it | pineapple → acoustic guitar | 3 | 0/0 | — | 0.0097 | 0.3102307617664337 | 0.18331396142259948 |
| layer_sweep | Gemma 4 E2B-it | Gemma 4 E4B-it | laptop → espresso | 3 | 0/0 | — | 0.0024 | 0.2799815833568573 | 0.360394649535891 |
| layer_sweep | Gemma 4 E2B-it | Gemma 4 E4B-it | volcano → rocking chair | 3 | 0/0 | — | 0.0071 | 0.25103679299354553 | 0.15458998600990645 |
| layer_sweep | Gemma 4 E2B-it | Gemma 4 E4B-it | soccer ball → school bus | 3 | 0/0 | — | 0.0017 | 0.3302059769630432 | 0.49468017983054124 |
| layer_sweep | InternVL3.5-2B-HF | Qwen3.5-4B | goldfish → monarch butterfly | 11 | 0/26 | 0.000 | 0.0000 | 0.5459592938423157 | 0.5182704788686402 |
| layer_sweep | InternVL3.5-2B-HF | Qwen3.5-4B | pineapple → acoustic guitar | 11 | 0/27 | 0.000 | 0.0000 | 0.7340016961097717 | 0.5759608802254266 |
| layer_sweep | InternVL3.5-2B-HF | Qwen3.5-4B | laptop → espresso | 11 | 0/15 | 0.000 | 0.0000 | 0.7877373099327087 | 0.617269291673402 |
| layer_sweep | InternVL3.5-2B-HF | Qwen3.5-4B | volcano → rocking chair | 11 | 3/27 | 0.111 | 0.0001 | 0.8097611665725708 | 0.613081035087791 |
| layer_sweep | InternVL3.5-2B-HF | Qwen3.5-4B | soccer ball → school bus | 11 | 0/18 | 0.000 | 0.0000 | 0.7319856286048889 | 0.49472555991091377 |
| layer_sweep | Gemma 4 E2B-it | Gemma 4 E4B-it | goldfish → monarch butterfly | 15 | 30/30 | 1.000 | 0.0522 | 1.0 | 1.0 |
| layer_sweep | Gemma 4 E2B-it | Gemma 4 E4B-it | pineapple → acoustic guitar | 15 | 30/30 | 1.000 | 0.0813 | 1.0 | 1.0 |
| layer_sweep | Gemma 4 E2B-it | Gemma 4 E4B-it | laptop → espresso | 15 | 30/30 | 1.000 | 0.0304 | 1.0 | 0.9999999999999998 |
| layer_sweep | Gemma 4 E2B-it | Gemma 4 E4B-it | volcano → rocking chair | 15 | 30/30 | 1.000 | 0.0579 | 1.0 | 0.9999999999999998 |
| layer_sweep | Gemma 4 E2B-it | Gemma 4 E4B-it | soccer ball → school bus | 15 | 30/30 | 1.000 | 0.0250 | 1.0 | 1.0 |
| layer_sweep | InternVL3.5-2B-HF | InternVL3.5-4B-HF | goldfish → monarch butterfly | 0 | 0/0 | — | 0.0030 | 0.2928559482097626 | 0.32510842195311723 |
| layer_sweep | InternVL3.5-2B-HF | InternVL3.5-4B-HF | pineapple → acoustic guitar | 0 | 0/0 | — | -0.0002 | 0.2690223753452301 | 0.32033556378029693 |
| layer_sweep | InternVL3.5-2B-HF | InternVL3.5-4B-HF | laptop → espresso | 0 | 0/0 | — | 0.0005 | 0.2475278526544571 | 0.2245995835355451 |
| layer_sweep | InternVL3.5-2B-HF | InternVL3.5-4B-HF | volcano → rocking chair | 0 | 0/0 | — | 0.0072 | 0.2511371672153473 | 0.24658335544221627 |
| layer_sweep | InternVL3.5-2B-HF | InternVL3.5-4B-HF | soccer ball → school bus | 0 | 0/0 | — | 0.0019 | 0.24161365628242493 | 0.22817078492477103 |
| layer_sweep | InternVL3.5-2B-HF | InternVL3.5-4B-HF | goldfish → monarch butterfly | 17 | 27/30 | 0.900 | 0.0353 | 0.9522922039031982 | 0.8472959168107534 |
| layer_sweep | InternVL3.5-2B-HF | InternVL3.5-4B-HF | pineapple → acoustic guitar | 17 | 9/30 | 0.300 | 0.0251 | 0.9646772742271423 | 0.9135339474589245 |
| layer_sweep | InternVL3.5-2B-HF | InternVL3.5-4B-HF | laptop → espresso | 17 | 6/29 | 0.207 | 0.0085 | 0.9473710060119629 | 0.826919564852216 |
| layer_sweep | InternVL3.5-2B-HF | InternVL3.5-4B-HF | volcano → rocking chair | 17 | 22/30 | 0.733 | 0.0482 | 0.9606459140777588 | 0.8865313362402674 |
| layer_sweep | InternVL3.5-2B-HF | InternVL3.5-4B-HF | soccer ball → school bus | 17 | 2/30 | 0.067 | 0.0096 | 0.9348904490470886 | 0.7669249273592251 |
| layer_sweep | Qwen3.5-2B | Gemma 4 E4B-it | goldfish → monarch butterfly | 5 | 0/0 | — | 0.0023 | 0.2426636666059494 | 0.2556092837394244 |
| layer_sweep | Qwen3.5-2B | Gemma 4 E4B-it | pineapple → acoustic guitar | 5 | 0/0 | — | 0.0208 | 0.27651742100715637 | 0.19895100390191964 |
| layer_sweep | Qwen3.5-2B | Gemma 4 E4B-it | laptop → espresso | 5 | 0/0 | — | 0.0042 | 0.2972452640533447 | 0.38036493848575986 |
| layer_sweep | Qwen3.5-2B | Gemma 4 E4B-it | volcano → rocking chair | 5 | 0/0 | — | 0.0153 | 0.2568211853504181 | 0.19599506711922232 |
| layer_sweep | Qwen3.5-2B | Gemma 4 E4B-it | soccer ball → school bus | 5 | 0/0 | — | 0.0040 | 0.3154867887496948 | 0.4252781840152666 |
| layer_sweep | InternVL3.5-2B-HF | InternVL3.5-4B-HF | goldfish → monarch butterfly | 11 | 11/26 | 0.423 | 0.0228 | 0.6780881881713867 | 0.6154996164205451 |
| layer_sweep | InternVL3.5-2B-HF | InternVL3.5-4B-HF | pineapple → acoustic guitar | 11 | 0/25 | 0.000 | 0.0222 | 0.8278293013572693 | 0.7641901156975012 |
| layer_sweep | InternVL3.5-2B-HF | InternVL3.5-4B-HF | laptop → espresso | 11 | 2/13 | 0.154 | 0.0090 | 0.8291575312614441 | 0.681478656157448 |
| layer_sweep | InternVL3.5-2B-HF | InternVL3.5-4B-HF | volcano → rocking chair | 11 | 17/25 | 0.680 | 0.0552 | 0.8307161927223206 | 0.7295705654723655 |
| layer_sweep | InternVL3.5-2B-HF | InternVL3.5-4B-HF | soccer ball → school bus | 11 | 5/17 | 0.294 | 0.0138 | 0.7977343201637268 | 0.6831224778811863 |
| layer_sweep | InternVL3.5-2B-HF | InternVL3.5-4B-HF | goldfish → monarch butterfly | 5 | 0/0 | — | 0.0053 | 0.48900601267814636 | 0.43393991078505123 |
| layer_sweep | InternVL3.5-2B-HF | InternVL3.5-4B-HF | pineapple → acoustic guitar | 5 | 0/0 | — | 0.0059 | 0.6870188117027283 | 0.6123517367343468 |
| layer_sweep | InternVL3.5-2B-HF | InternVL3.5-4B-HF | laptop → espresso | 5 | 0/0 | — | 0.0025 | 0.49333611130714417 | 0.3557561135452521 |
| layer_sweep | InternVL3.5-2B-HF | InternVL3.5-4B-HF | volcano → rocking chair | 5 | 0/0 | — | 0.0216 | 0.507908046245575 | 0.4135454737428045 |
| layer_sweep | InternVL3.5-2B-HF | InternVL3.5-4B-HF | soccer ball → school bus | 5 | 0/0 | — | 0.0051 | 0.5399320721626282 | 0.5118596512325415 |
| layer_sweep | Qwen3.5-2B | Qwen3.5-4B | goldfish → monarch butterfly | 11 | 23/28 | 0.821 | 0.0002 | 0.250129371881485 | 0.33894246953887097 |
| layer_sweep | Qwen3.5-2B | Qwen3.5-4B | pineapple → acoustic guitar | 11 | 28/29 | 0.966 | 0.0001 | 0.46884027123451233 | 0.46913096608397015 |
| layer_sweep | Qwen3.5-2B | Qwen3.5-4B | laptop → espresso | 11 | 8/10 | 0.800 | 0.0000 | 0.4357205927371979 | 0.4310647607130779 |
| layer_sweep | Qwen3.5-2B | Qwen3.5-4B | volcano → rocking chair | 11 | 28/28 | 1.000 | 0.0002 | 0.39566749334335327 | 0.3192646431462955 |
| layer_sweep | Qwen3.5-2B | Qwen3.5-4B | soccer ball → school bus | 11 | 16/17 | 0.941 | 0.0001 | 0.3895244598388672 | 0.31863001006306446 |
| layer_sweep | InternVL3.5-2B-HF | Qwen3.5-4B | goldfish → monarch butterfly | 0 | 0/0 | — | 0.0000 | 0.25367438793182373 | 0.28391318625076595 |
| layer_sweep | InternVL3.5-2B-HF | Qwen3.5-4B | pineapple → acoustic guitar | 0 | 0/0 | — | -0.0000 | 0.3917476236820221 | 0.41325985099476825 |
| layer_sweep | InternVL3.5-2B-HF | Qwen3.5-4B | laptop → espresso | 0 | 0/0 | — | 0.0000 | 0.33706390857696533 | 0.3754306562751937 |
| layer_sweep | InternVL3.5-2B-HF | Qwen3.5-4B | volcano → rocking chair | 0 | 0/0 | — | 0.0000 | 0.3731740117073059 | 0.31110689710298306 |
| layer_sweep | InternVL3.5-2B-HF | Qwen3.5-4B | soccer ball → school bus | 0 | 0/0 | — | -0.0000 | 0.32016900181770325 | 0.20821988527977775 |
| layer_sweep | InternVL3.5-2B-HF | Qwen3.5-4B | goldfish → monarch butterfly | 5 | 0/0 | — | 0.0000 | 0.39535945653915405 | 0.4058986113387422 |
| layer_sweep | InternVL3.5-2B-HF | Qwen3.5-4B | pineapple → acoustic guitar | 5 | 0/0 | — | 0.0000 | 0.6389913558959961 | 0.5372533809108042 |
| layer_sweep | InternVL3.5-2B-HF | Qwen3.5-4B | laptop → espresso | 5 | 0/0 | — | 0.0000 | 0.49283668398857117 | 0.35508017391293195 |
| layer_sweep | InternVL3.5-2B-HF | Qwen3.5-4B | volcano → rocking chair | 5 | 0/0 | — | 0.0001 | 0.5687416195869446 | 0.4173486327697593 |
| layer_sweep | InternVL3.5-2B-HF | Qwen3.5-4B | soccer ball → school bus | 5 | 0/0 | — | 0.0000 | 0.5049216151237488 | 0.36591925172836576 |
| ablation | Qwen3.5-2B | Qwen3.5-4B | goldfish → monarch butterfly | 23 | 29/30 | 0.967 | 0.0002 | 0.9572506546974182 | 0.919183994272392 |
| ablation | Qwen3.5-2B | Qwen3.5-4B | pineapple → acoustic guitar | 23 | 16/29 | 0.552 | 0.0001 | 0.9611946940422058 | 0.9094387122705928 |
| ablation | Qwen3.5-2B | Qwen3.5-4B | laptop → espresso | 23 | 6/12 | 0.500 | 0.0000 | 0.9238079786300659 | 0.8302092751133474 |
| ablation | Qwen3.5-2B | Qwen3.5-4B | volcano → rocking chair | 23 | 30/30 | 1.000 | 0.0002 | 0.9635360836982727 | 0.895682610392392 |
| ablation | Qwen3.5-2B | Qwen3.5-4B | soccer ball → school bus | 23 | 10/25 | 0.400 | 0.0000 | 0.9427202939987183 | 0.8640513607699123 |
| ablation | Gemma 4 E2B-it | Gemma 4 E4B-it | goldfish → monarch butterfly | 15 | 30/30 | 1.000 | 0.0525 | 1.0 | 1.0 |
| ablation | Gemma 4 E2B-it | Gemma 4 E4B-it | pineapple → acoustic guitar | 15 | 30/30 | 1.000 | 0.0815 | 1.0 | 1.0 |
| ablation | Gemma 4 E2B-it | Gemma 4 E4B-it | laptop → espresso | 15 | 30/30 | 1.000 | 0.0305 | 1.0 | 0.9999999999999998 |
| ablation | Gemma 4 E2B-it | Gemma 4 E4B-it | volcano → rocking chair | 15 | 30/30 | 1.000 | 0.0579 | 1.0 | 0.9999999999999998 |
| ablation | Gemma 4 E2B-it | Gemma 4 E4B-it | soccer ball → school bus | 15 | 30/30 | 1.000 | 0.0249 | 1.0 | 1.0 |
| ablation | Qwen3.5-2B | Qwen3.5-4B | goldfish → monarch butterfly | 23 | 0/0 | — | 0.0001 | 0.9572506546974182 | 0.919183994272392 |
| ablation | Qwen3.5-2B | Qwen3.5-4B | pineapple → acoustic guitar | 23 | 0/0 | — | 0.0000 | 0.9611946940422058 | 0.9094387122705928 |
| ablation | Qwen3.5-2B | Qwen3.5-4B | laptop → espresso | 23 | 0/0 | — | -0.0000 | 0.9238079786300659 | 0.8302092751133474 |
| ablation | Qwen3.5-2B | Qwen3.5-4B | volcano → rocking chair | 23 | 0/0 | — | 0.0001 | 0.9635360836982727 | 0.895682610392392 |
| ablation | Qwen3.5-2B | Qwen3.5-4B | soccer ball → school bus | 23 | 0/0 | — | 0.0000 | 0.9427202939987183 | 0.8640513607699123 |
| ablation | Qwen3.5-2B | Qwen3.5-4B | goldfish → monarch butterfly | 23 | 0/0 | — | 0.0001 | 0.9572506546974182 | 0.919183994272392 |
| ablation | Qwen3.5-2B | Qwen3.5-4B | pineapple → acoustic guitar | 23 | 0/0 | — | 0.0000 | 0.9611946940422058 | 0.9094387122705928 |
| ablation | Qwen3.5-2B | Qwen3.5-4B | laptop → espresso | 23 | 0/0 | — | 0.0000 | 0.9238079786300659 | 0.8302092751133474 |
| ablation | Qwen3.5-2B | Qwen3.5-4B | volcano → rocking chair | 23 | 0/0 | — | 0.0001 | 0.9635360836982727 | 0.895682610392392 |
| ablation | Qwen3.5-2B | Qwen3.5-4B | soccer ball → school bus | 23 | 0/0 | — | 0.0000 | 0.9427202939987183 | 0.8640513607699123 |
| ablation | Gemma 4 E2B-it | Gemma 4 E4B-it | goldfish → monarch butterfly | 15 | 30/30 | 1.000 | 0.0360 | 1.0 | 1.0 |
| ablation | Gemma 4 E2B-it | Gemma 4 E4B-it | pineapple → acoustic guitar | 15 | 30/30 | 1.000 | 0.0600 | 1.0 | 1.0 |
| ablation | Gemma 4 E2B-it | Gemma 4 E4B-it | laptop → espresso | 15 | 29/30 | 0.967 | 0.0209 | 1.0 | 0.9999999999999998 |
| ablation | Gemma 4 E2B-it | Gemma 4 E4B-it | volcano → rocking chair | 15 | 30/30 | 1.000 | 0.0413 | 1.0 | 0.9999999999999998 |
| ablation | Gemma 4 E2B-it | Gemma 4 E4B-it | soccer ball → school bus | 15 | 29/30 | 0.967 | 0.0172 | 1.0 | 1.0 |
| ablation | Gemma 4 E2B-it | Gemma 4 E4B-it | goldfish → monarch butterfly | 15 | 30/30 | 1.000 | 0.1429 | 1.0 | 1.0 |
| ablation | Gemma 4 E2B-it | Gemma 4 E4B-it | pineapple → acoustic guitar | 15 | 30/30 | 1.000 | 0.1396 | 1.0 | 1.0 |
| ablation | Gemma 4 E2B-it | Gemma 4 E4B-it | laptop → espresso | 15 | 30/30 | 1.000 | 0.0926 | 1.0 | 0.9999999999999998 |
| ablation | Gemma 4 E2B-it | Gemma 4 E4B-it | volcano → rocking chair | 15 | 30/30 | 1.000 | 0.1142 | 1.0 | 0.9999999999999998 |
| ablation | Gemma 4 E2B-it | Gemma 4 E4B-it | soccer ball → school bus | 15 | 30/30 | 1.000 | 0.0639 | 1.0 | 1.0 |
| ablation | Qwen3.5-2B | Qwen3.5-4B | goldfish → monarch butterfly | 23 | 6/11 | 0.545 | 0.0002 | 0.9572506546974182 | 0.919183994272392 |
| ablation | Qwen3.5-2B | Qwen3.5-4B | pineapple → acoustic guitar | 23 | 5/12 | 0.417 | 0.0001 | 0.9611946940422058 | 0.9094387122705928 |
| ablation | Qwen3.5-2B | Qwen3.5-4B | laptop → espresso | 23 | 2/7 | 0.286 | 0.0000 | 0.9238079786300659 | 0.8302092751133474 |
| ablation | Qwen3.5-2B | Qwen3.5-4B | volcano → rocking chair | 23 | 11/17 | 0.647 | 0.0001 | 0.9635360836982727 | 0.895682610392392 |
| ablation | Qwen3.5-2B | Qwen3.5-4B | soccer ball → school bus | 23 | 5/17 | 0.294 | 0.0000 | 0.9427202939987183 | 0.8640513607699123 |
| ablation | Gemma 4 E2B-it | Gemma 4 E4B-it | goldfish → monarch butterfly | 15 | 2/5 | 0.400 | 0.0372 | 1.0 | 1.0 |
| ablation | Gemma 4 E2B-it | Gemma 4 E4B-it | pineapple → acoustic guitar | 15 | 5/7 | 0.714 | 0.0518 | 1.0 | 1.0 |
| ablation | Gemma 4 E2B-it | Gemma 4 E4B-it | laptop → espresso | 15 | 1/1 | 1.000 | 0.0280 | 1.0 | 0.9999999999999998 |
| ablation | Gemma 4 E2B-it | Gemma 4 E4B-it | volcano → rocking chair | 15 | 1/1 | 1.000 | 0.0066 | 1.0 | 0.9999999999999998 |
| ablation | Gemma 4 E2B-it | Gemma 4 E4B-it | soccer ball → school bus | 15 | 1/1 | 1.000 | 0.0033 | 1.0 | 1.0 |
| ablation | Qwen3.5-2B | Qwen3.5-4B | goldfish → monarch butterfly | 23 | 27/30 | 0.900 | 0.0002 | 0.9572506546974182 | 0.919183994272392 |
| ablation | Qwen3.5-2B | Qwen3.5-4B | pineapple → acoustic guitar | 23 | 2/3 | 0.667 | 0.0000 | 0.9611946940422058 | 0.9094387122705928 |
| ablation | Qwen3.5-2B | Qwen3.5-4B | laptop → espresso | 23 | 1/2 | 0.500 | 0.0000 | 0.9238079786300659 | 0.8302092751133474 |
| ablation | Qwen3.5-2B | Qwen3.5-4B | volcano → rocking chair | 23 | 27/29 | 0.931 | 0.0001 | 0.9635360836982727 | 0.895682610392392 |
| ablation | Qwen3.5-2B | Qwen3.5-4B | soccer ball → school bus | 23 | 3/10 | 0.300 | 0.0000 | 0.9427202939987183 | 0.8640513607699123 |
| ablation | Gemma 4 E2B-it | Gemma 4 E4B-it | goldfish → monarch butterfly | 15 | 1/2 | 0.500 | 0.0250 | 1.0 | 1.0 |
| ablation | Gemma 4 E2B-it | Gemma 4 E4B-it | pineapple → acoustic guitar | 15 | 0/1 | 0.000 | 0.0055 | 1.0 | 1.0 |
| ablation | Gemma 4 E2B-it | Gemma 4 E4B-it | laptop → espresso | 15 | 0/0 | — | 0.0212 | 1.0 | 0.9999999999999998 |
| ablation | Gemma 4 E2B-it | Gemma 4 E4B-it | volcano → rocking chair | 15 | 0/0 | — | -0.0077 | 1.0 | 0.9999999999999998 |
| ablation | Gemma 4 E2B-it | Gemma 4 E4B-it | soccer ball → school bus | 15 | 0/0 | — | -0.0027 | 1.0 | 1.0 |
| reverse_direction | Gemma 4 E2B-it | Gemma 4 E4B-it | monarch butterfly → goldfish | 15 | 30/30 | 1.000 | 0.0504 | 1.0 | 1.0 |
| reverse_direction | Gemma 4 E2B-it | Gemma 4 E4B-it | acoustic guitar → pineapple | 15 | 30/30 | 1.000 | 0.0860 | 1.0 | 1.0 |
| reverse_direction | Gemma 4 E2B-it | Gemma 4 E4B-it | espresso → laptop | 15 | 28/28 | 1.000 | 0.0345 | 1.0 | 0.9999999999999998 |
| reverse_direction | Gemma 4 E2B-it | Gemma 4 E4B-it | rocking chair → volcano | 15 | 30/30 | 1.000 | 0.0585 | 1.0 | 0.9999999999999998 |
| reverse_direction | Gemma 4 E2B-it | Gemma 4 E4B-it | school bus → soccer ball | 15 | 30/30 | 1.000 | 0.0257 | 1.0 | 1.0 |
| reverse_direction | Gemma 4 E2B-it | Gemma 4 26B-A4B-it | monarch butterfly → goldfish | 15 | 0/30 | 0.000 | 0.0075 | 0.9571262001991272 | 0.9026865443320575 |
| reverse_direction | Gemma 4 E2B-it | Gemma 4 26B-A4B-it | acoustic guitar → pineapple | 15 | 1/30 | 0.033 | 0.0139 | 0.932298481464386 | 0.8886508062939197 |
| reverse_direction | Gemma 4 E2B-it | Gemma 4 26B-A4B-it | espresso → laptop | 15 | 1/28 | 0.036 | 0.0038 | 0.9033745527267456 | 0.8599638076943454 |
| reverse_direction | Gemma 4 E2B-it | Gemma 4 26B-A4B-it | rocking chair → volcano | 15 | 1/30 | 0.033 | 0.0069 | 0.9484579563140869 | 0.888266571553616 |
| reverse_direction | Gemma 4 E2B-it | Gemma 4 26B-A4B-it | school bus → soccer ball | 15 | 0/30 | 0.000 | 0.0030 | 0.926635205745697 | 0.8937779149251907 |
| reverse_direction | InternVL3.5-2B-HF | InternVL3.5-4B-HF | monarch butterfly → goldfish | 23 | 2/30 | 0.067 | 0.0275 | 0.9660874009132385 | 0.9137480511021122 |
| reverse_direction | InternVL3.5-2B-HF | InternVL3.5-4B-HF | acoustic guitar → pineapple | 23 | 1/30 | 0.033 | 0.0246 | 0.9570565819740295 | 0.9058564879589585 |
| reverse_direction | InternVL3.5-2B-HF | InternVL3.5-4B-HF | espresso → laptop | 23 | 2/30 | 0.067 | 0.0207 | 0.9477024674415588 | 0.8925706957395585 |
| reverse_direction | InternVL3.5-2B-HF | InternVL3.5-4B-HF | rocking chair → volcano | 23 | 2/30 | 0.067 | 0.0187 | 0.974137544631958 | 0.9261488980194571 |
| reverse_direction | InternVL3.5-2B-HF | InternVL3.5-4B-HF | school bus → soccer ball | 23 | 0/29 | 0.000 | 0.0077 | 0.9472202062606812 | 0.8764245652236676 |
| reverse_direction | Qwen3.5-2B | Gemma 4 E4B-it | monarch butterfly → goldfish | 23 | 0/30 | 0.000 | 0.0111 | 0.8509442806243896 | 0.7120382811920831 |
| reverse_direction | Qwen3.5-2B | Gemma 4 E4B-it | acoustic guitar → pineapple | 23 | 0/27 | 0.000 | 0.0075 | 0.7642777562141418 | 0.5277742813395394 |
| reverse_direction | Qwen3.5-2B | Gemma 4 E4B-it | espresso → laptop | 23 | 0/21 | 0.000 | 0.0060 | 0.7853063941001892 | 0.6010451529782691 |
| reverse_direction | Qwen3.5-2B | Gemma 4 E4B-it | rocking chair → volcano | 23 | 0/29 | 0.000 | 0.0033 | 0.8427858948707581 | 0.6248395483169762 |
| reverse_direction | Qwen3.5-2B | Gemma 4 E4B-it | school bus → soccer ball | 23 | 0/9 | 0.000 | 0.0026 | 0.826906144618988 | 0.6815021936859391 |
| reverse_direction | Gemma 4 E2B-it | InternVL3.5-4B-HF | monarch butterfly → goldfish | 15 | 0/30 | 0.000 | 0.0134 | 0.9418367147445679 | 0.8409391763459584 |
| reverse_direction | Gemma 4 E2B-it | InternVL3.5-4B-HF | acoustic guitar → pineapple | 15 | 0/30 | 0.000 | 0.0135 | 0.9085941910743713 | 0.750940208052508 |
| reverse_direction | Gemma 4 E2B-it | InternVL3.5-4B-HF | espresso → laptop | 15 | 1/25 | 0.040 | 0.0157 | 0.9026076793670654 | 0.738224495456555 |
| reverse_direction | Gemma 4 E2B-it | InternVL3.5-4B-HF | rocking chair → volcano | 15 | 0/30 | 0.000 | 0.0100 | 0.928453803062439 | 0.7981106763834439 |
| reverse_direction | Gemma 4 E2B-it | InternVL3.5-4B-HF | school bus → soccer ball | 15 | 0/30 | 0.000 | 0.0027 | 0.8687721490859985 | 0.7181038222246017 |
| reverse_direction | Qwen3.5-2B | Qwen3.5-4B | monarch butterfly → goldfish | 23 | 15/30 | 0.500 | 0.0001 | 0.9572506546974182 | 0.919183994272392 |
| reverse_direction | Qwen3.5-2B | Qwen3.5-4B | acoustic guitar → pineapple | 23 | 12/27 | 0.444 | 0.0001 | 0.961194634437561 | 0.9094387122705928 |
| reverse_direction | Qwen3.5-2B | Qwen3.5-4B | espresso → laptop | 23 | 10/22 | 0.455 | 0.0001 | 0.9238079786300659 | 0.8302092751133474 |
| reverse_direction | Qwen3.5-2B | Qwen3.5-4B | rocking chair → volcano | 23 | 6/29 | 0.207 | 0.0001 | 0.9635362029075623 | 0.895682610392392 |
| reverse_direction | Qwen3.5-2B | Qwen3.5-4B | school bus → soccer ball | 23 | 2/10 | 0.200 | 0.0001 | 0.9427204132080078 | 0.8640513607699123 |
| reverse_direction | Qwen3.5-2B | Qwen3.5-9B | monarch butterfly → goldfish | 23 | 1/30 | 0.033 | 0.0000 | 0.9408534169197083 | 0.8766009203812413 |
| reverse_direction | Qwen3.5-2B | Qwen3.5-9B | acoustic guitar → pineapple | 23 | 0/27 | 0.000 | 0.0000 | 0.914558470249176 | 0.8489998390409518 |
| reverse_direction | Qwen3.5-2B | Qwen3.5-9B | espresso → laptop | 23 | 0/21 | 0.000 | 0.0000 | 0.8376241326332092 | 0.6445863072178778 |
| reverse_direction | Qwen3.5-2B | Qwen3.5-9B | rocking chair → volcano | 23 | 0/29 | 0.000 | 0.0000 | 0.9071047902107239 | 0.7958412166731171 |
| reverse_direction | Qwen3.5-2B | Qwen3.5-9B | school bus → soccer ball | 23 | 0/9 | 0.000 | 0.0000 | 0.8854108452796936 | 0.7975380583077056 |
| reverse_direction | InternVL3.5-2B-HF | Gemma 4 E4B-it | monarch butterfly → goldfish | 23 | 0/30 | 0.000 | 0.0109 | 0.9368488788604736 | 0.8553114802759518 |
| reverse_direction | InternVL3.5-2B-HF | Gemma 4 E4B-it | acoustic guitar → pineapple | 23 | 0/30 | 0.000 | 0.0069 | 0.8564447164535522 | 0.6547870531027303 |
| reverse_direction | InternVL3.5-2B-HF | Gemma 4 E4B-it | espresso → laptop | 23 | 0/30 | 0.000 | 0.0050 | 0.9002618789672852 | 0.7505661835450219 |
| reverse_direction | InternVL3.5-2B-HF | Gemma 4 E4B-it | rocking chair → volcano | 23 | 0/30 | 0.000 | 0.0042 | 0.9298343658447266 | 0.7992798860584072 |
| reverse_direction | InternVL3.5-2B-HF | Gemma 4 E4B-it | school bus → soccer ball | 23 | 0/30 | 0.000 | 0.0030 | 0.8798831701278687 | 0.728837737748348 |
| reverse_direction | Qwen3.5-2B | InternVL3.5-4B-HF | monarch butterfly → goldfish | 23 | 0/30 | 0.000 | 0.0119 | 0.8760660290718079 | 0.7532517662009103 |
| reverse_direction | Qwen3.5-2B | InternVL3.5-4B-HF | acoustic guitar → pineapple | 23 | 0/27 | 0.000 | 0.0111 | 0.8807937502861023 | 0.7965858369976894 |
| reverse_direction | Qwen3.5-2B | InternVL3.5-4B-HF | espresso → laptop | 23 | 0/20 | 0.000 | 0.0111 | 0.825928270816803 | 0.6303236516918393 |
| reverse_direction | Qwen3.5-2B | InternVL3.5-4B-HF | rocking chair → volcano | 23 | 0/29 | 0.000 | 0.0088 | 0.863957405090332 | 0.7222413740967455 |
| reverse_direction | Qwen3.5-2B | InternVL3.5-4B-HF | school bus → soccer ball | 23 | 0/9 | 0.000 | 0.0030 | 0.8293028473854065 | 0.7120069379126106 |
| reverse_direction | InternVL3.5-2B-HF | Qwen3.5-4B | monarch butterfly → goldfish | 23 | 0/30 | 0.000 | 0.0000 | 0.9007353782653809 | 0.8068765031340197 |
| reverse_direction | InternVL3.5-2B-HF | Qwen3.5-4B | acoustic guitar → pineapple | 23 | 0/30 | 0.000 | 0.0000 | 0.9254666566848755 | 0.8527184915490558 |
| reverse_direction | InternVL3.5-2B-HF | Qwen3.5-4B | espresso → laptop | 23 | 0/30 | 0.000 | 0.0000 | 0.8917120099067688 | 0.7328102200688649 |
| reverse_direction | InternVL3.5-2B-HF | Qwen3.5-4B | rocking chair → volcano | 23 | 0/30 | 0.000 | 0.0000 | 0.8988474607467651 | 0.7617233282285629 |
| reverse_direction | InternVL3.5-2B-HF | Qwen3.5-4B | school bus → soccer ball | 23 | 0/30 | 0.000 | 0.0000 | 0.8831189870834351 | 0.7411675301760164 |
| reverse_direction | InternVL3.5-2B-HF | InternVL3.5-8B-HF | monarch butterfly → goldfish | 23 | 3/30 | 0.100 | 0.0296 | 0.9679560661315918 | 0.9327099421440115 |
| reverse_direction | InternVL3.5-2B-HF | InternVL3.5-8B-HF | acoustic guitar → pineapple | 23 | 0/30 | 0.000 | 0.0356 | 0.9790650010108948 | 0.9532871806750695 |
| reverse_direction | InternVL3.5-2B-HF | InternVL3.5-8B-HF | espresso → laptop | 23 | 5/30 | 0.167 | 0.0215 | 0.9495866298675537 | 0.9061634809251022 |
| reverse_direction | InternVL3.5-2B-HF | InternVL3.5-8B-HF | rocking chair → volcano | 23 | 0/30 | 0.000 | 0.0208 | 0.9691880941390991 | 0.912080013164289 |
| reverse_direction | InternVL3.5-2B-HF | InternVL3.5-8B-HF | school bus → soccer ball | 23 | 0/30 | 0.000 | 0.0069 | 0.9484160542488098 | 0.8717490832323199 |
| reverse_direction | Gemma 4 E2B-it | Qwen3.5-4B | monarch butterfly → goldfish | 15 | 0/30 | 0.000 | 0.0000 | 0.8674367070198059 | 0.7460722647620969 |
| reverse_direction | Gemma 4 E2B-it | Qwen3.5-4B | acoustic guitar → pineapple | 15 | 0/30 | 0.000 | 0.0000 | 0.795652449131012 | 0.5024936072839901 |
| reverse_direction | Gemma 4 E2B-it | Qwen3.5-4B | espresso → laptop | 15 | 0/27 | 0.000 | 0.0000 | 0.8433840274810791 | 0.6087014194376508 |
| reverse_direction | Gemma 4 E2B-it | Qwen3.5-4B | rocking chair → volcano | 15 | 0/30 | 0.000 | 0.0000 | 0.866771936416626 | 0.6194269359076255 |
| reverse_direction | Gemma 4 E2B-it | Qwen3.5-4B | school bus → soccer ball | 15 | 0/30 | 0.000 | -0.0000 | 0.8193195462226868 | 0.5438602368238711 |
| multiple_proxy | Qwen3.5-2B+Qwen3.5-4B | Gemma 4 31B-it | soccer ball → school bus | -1 | 0/28 | 0.000 | 0.0019 | 0.8606749176979065 | 0.7020294197640059 |
| multiple_proxy | Qwen3.5-2B+Qwen3.5-4B | Gemma 4 31B-it | volcano → rocking chair | -1 | 1/30 | 0.033 | 0.0059 | 0.8711766600608826 | 0.6661601127496264 |
| multiple_proxy | Qwen3.5-2B+Qwen3.5-4B | Gemma 4 31B-it | laptop → espresso | -1 | 0/18 | 0.000 | 0.0054 | 0.8005383610725403 | 0.6520544243448572 |
| multiple_proxy | Qwen3.5-2B+Qwen3.5-4B | Gemma 4 31B-it | goldfish → monarch butterfly | -1 | 1/30 | 0.033 | 0.0069 | 0.9046216905117035 | 0.7938646590605118 |
| multiple_proxy | Qwen3.5-2B+Qwen3.5-4B | Gemma 4 31B-it | pineapple → acoustic guitar | -1 | 0/30 | 0.000 | 0.0049 | 0.7392197549343109 | 0.6062601153833171 |
| multiple_proxy | InternVL3.5-2B-HF+InternVL3.5-4B-HF+InternVL3.5-8B-HF | Qwen3.5-27B | soccer ball → school bus | -1 | 0/30 | 0.000 | 0.0000 | 0.8359290162722269 | 0.709861782000584 |
| multiple_proxy | InternVL3.5-2B-HF+InternVL3.5-4B-HF+InternVL3.5-8B-HF | Qwen3.5-27B | volcano → rocking chair | -1 | 7/30 | 0.233 | 0.0002 | 0.8742004235585531 | 0.7322711950030897 |
| multiple_proxy | InternVL3.5-2B-HF+InternVL3.5-4B-HF+InternVL3.5-8B-HF | Qwen3.5-27B | laptop → espresso | -1 | 0/30 | 0.000 | 0.0000 | 0.8939377864201864 | 0.7655706341029286 |
| multiple_proxy | InternVL3.5-2B-HF+InternVL3.5-4B-HF+InternVL3.5-8B-HF | Qwen3.5-27B | goldfish → monarch butterfly | -1 | 7/30 | 0.233 | 0.0002 | 0.8934812744458517 | 0.8033871997020733 |
| multiple_proxy | InternVL3.5-2B-HF+InternVL3.5-4B-HF+InternVL3.5-8B-HF | Qwen3.5-27B | pineapple → acoustic guitar | -1 | 0/30 | 0.000 | 0.0001 | 0.8707438707351685 | 0.7897291926374329 |
| multiple_proxy | Qwen3.5-2B+Qwen3.5-4B+Qwen3.5-9B | Gemma 4 31B-it | soccer ball → school bus | -1 | 0/27 | 0.000 | 0.0023 | 0.8588840564092001 | 0.6848289132868436 |
| multiple_proxy | Qwen3.5-2B+Qwen3.5-4B+Qwen3.5-9B | Gemma 4 31B-it | volcano → rocking chair | -1 | 2/30 | 0.067 | 0.0067 | 0.8743533690770467 | 0.6661754404374015 |
| multiple_proxy | Qwen3.5-2B+Qwen3.5-4B+Qwen3.5-9B | Gemma 4 31B-it | laptop → espresso | -1 | 1/18 | 0.056 | 0.0058 | 0.8050577243169149 | 0.669704642904532 |
| multiple_proxy | Qwen3.5-2B+Qwen3.5-4B+Qwen3.5-9B | Gemma 4 31B-it | goldfish → monarch butterfly | -1 | 6/30 | 0.200 | 0.0079 | 0.9000653624534607 | 0.7880198034091732 |
| multiple_proxy | Qwen3.5-2B+Qwen3.5-4B+Qwen3.5-9B | Gemma 4 31B-it | pineapple → acoustic guitar | -1 | 0/29 | 0.000 | 0.0049 | 0.746137797832489 | 0.6223480708193617 |
| multiple_proxy | Gemma 4 E2B-it+Gemma 4 E4B-it+Gemma 4 26B-A4B-it | InternVL3.5-14B-HF | laptop → espresso | -1 | 1/1 | 1.000 | 0.0384 | 0.8875456055005392 | 0.729556611561804 |
| multiple_proxy | Gemma 4 E2B-it+Gemma 4 E4B-it+Gemma 4 26B-A4B-it | InternVL3.5-14B-HF | goldfish → monarch butterfly | -1 | 11/30 | 0.367 | 0.1060 | 0.9477190772692362 | 0.8454432403537346 |
| multiple_proxy | Gemma 4 E2B-it+Gemma 4 E4B-it+Gemma 4 26B-A4B-it | InternVL3.5-14B-HF | pineapple → acoustic guitar | -1 | 0/0 | — | 0.0578 | 0.8798856735229492 | 0.7351730701800697 |
| multiple_proxy | InternVL3.5-2B-HF+Gemma 4 E2B-it | Qwen3.5-27B | soccer ball → school bus | -1 | 0/30 | 0.000 | 0.0000 | 0.824589729309082 | 0.684278802261931 |
| multiple_proxy | InternVL3.5-2B-HF+Gemma 4 E2B-it | Qwen3.5-27B | volcano → rocking chair | -1 | 6/30 | 0.200 | 0.0002 | 0.8587621748447418 | 0.6802764263963852 |
| multiple_proxy | InternVL3.5-2B-HF+Gemma 4 E2B-it | Qwen3.5-27B | laptop → espresso | -1 | 0/30 | 0.000 | 0.0000 | 0.8600495755672455 | 0.7200651427248674 |
| multiple_proxy | InternVL3.5-2B-HF+Gemma 4 E2B-it | Qwen3.5-27B | goldfish → monarch butterfly | -1 | 4/30 | 0.133 | 0.0002 | 0.8834238946437836 | 0.7929058802844808 |
| multiple_proxy | InternVL3.5-2B-HF+Gemma 4 E2B-it | Qwen3.5-27B | pineapple → acoustic guitar | -1 | 0/30 | 0.000 | 0.0001 | 0.8114610016345978 | 0.6726774290518637 |
| multiple_proxy | Gemma 4 E2B-it | InternVL3.5-14B-HF | soccer ball → school bus | -1 | 0/30 | 0.000 | 0.0277 | 0.8577516674995422 | 0.7194677716847624 |
| multiple_proxy | Gemma 4 E2B-it | InternVL3.5-14B-HF | volcano → rocking chair | -1 | 7/30 | 0.233 | 0.1395 | 0.9385977387428284 | 0.8336775680851626 |
| multiple_proxy | Gemma 4 E2B-it | InternVL3.5-14B-HF | laptop → espresso | -1 | 0/30 | 0.000 | 0.0274 | 0.9043950438499451 | 0.7338813122233027 |
| multiple_proxy | Gemma 4 E2B-it | InternVL3.5-14B-HF | goldfish → monarch butterfly | -1 | 5/30 | 0.167 | 0.0826 | 0.9454782605171204 | 0.8397853935152674 |
| multiple_proxy | Gemma 4 E2B-it | InternVL3.5-14B-HF | pineapple → acoustic guitar | -1 | 0/30 | 0.000 | 0.0501 | 0.8963858485221863 | 0.7134193890229025 |
| multiple_proxy | InternVL3.5-2B-HF+InternVL3.5-4B-HF | Qwen3.5-27B | soccer ball → school bus | -1 | 0/30 | 0.000 | 0.0000 | 0.8301553130149841 | 0.7044730256129146 |
| multiple_proxy | InternVL3.5-2B-HF+InternVL3.5-4B-HF | Qwen3.5-27B | volcano → rocking chair | -1 | 3/30 | 0.100 | 0.0002 | 0.8736592233181 | 0.7311934063017986 |
| multiple_proxy | InternVL3.5-2B-HF+InternVL3.5-4B-HF | Qwen3.5-27B | laptop → espresso | -1 | 0/30 | 0.000 | 0.0000 | 0.889429360628128 | 0.7602150370864863 |
| multiple_proxy | InternVL3.5-2B-HF+InternVL3.5-4B-HF | Qwen3.5-27B | goldfish → monarch butterfly | -1 | 2/30 | 0.067 | 0.0002 | 0.8902992904186249 | 0.8020205091951795 |
| multiple_proxy | InternVL3.5-2B-HF+InternVL3.5-4B-HF | Qwen3.5-27B | pineapple → acoustic guitar | -1 | 0/30 | 0.000 | 0.0001 | 0.8736070096492767 | 0.7974979155705066 |
| multiple_proxy | InternVL3.5-2B-HF | Qwen3.5-27B | soccer ball → school bus | -1 | 0/30 | 0.000 | 0.0000 | 0.8389680981636047 | 0.7114510027230642 |
| multiple_proxy | InternVL3.5-2B-HF | Qwen3.5-27B | volcano → rocking chair | -1 | 1/30 | 0.033 | 0.0002 | 0.8698771595954895 | 0.720650561470335 |
| multiple_proxy | InternVL3.5-2B-HF | Qwen3.5-27B | laptop → espresso | -1 | 0/30 | 0.000 | 0.0000 | 0.8748825788497925 | 0.7418001824713781 |
| multiple_proxy | InternVL3.5-2B-HF | Qwen3.5-27B | goldfish → monarch butterfly | -1 | 2/30 | 0.067 | 0.0001 | 0.9005240797996521 | 0.8211092290031945 |
| multiple_proxy | InternVL3.5-2B-HF | Qwen3.5-27B | pineapple → acoustic guitar | -1 | 0/30 | 0.000 | 0.0001 | 0.8722259998321533 | 0.8102690733655892 |
| multiple_proxy | Qwen3.5-2B | Gemma 4 31B-it | soccer ball → school bus | -1 | 0/27 | 0.000 | 0.0019 | 0.863053023815155 | 0.7534217120734114 |
| multiple_proxy | Qwen3.5-2B | Gemma 4 31B-it | volcano → rocking chair | -1 | 1/30 | 0.033 | 0.0057 | 0.8613731861114502 | 0.6741923024832523 |
| multiple_proxy | Qwen3.5-2B | Gemma 4 31B-it | laptop → espresso | -1 | 0/17 | 0.000 | 0.0052 | 0.7659106254577637 | 0.624802481169013 |
| multiple_proxy | Qwen3.5-2B | Gemma 4 31B-it | goldfish → monarch butterfly | -1 | 0/30 | 0.000 | 0.0064 | 0.9005722999572754 | 0.7864497759178861 |
| multiple_proxy | Qwen3.5-2B | Gemma 4 31B-it | pineapple → acoustic guitar | -1 | 0/30 | 0.000 | 0.0053 | 0.7164500951766968 | 0.5874369357472894 |
| multiple_proxy | Gemma 4 E2B-it+Gemma 4 E4B-it | InternVL3.5-14B-HF | soccer ball → school bus | -1 | 0/30 | 0.000 | 0.0275 | 0.8577516674995422 | 0.7194677716847624 |
| multiple_proxy | Gemma 4 E2B-it+Gemma 4 E4B-it | InternVL3.5-14B-HF | volcano → rocking chair | -1 | 9/30 | 0.300 | 0.1383 | 0.9385977387428284 | 0.8336775680851626 |
| multiple_proxy | Gemma 4 E2B-it+Gemma 4 E4B-it | InternVL3.5-14B-HF | laptop → espresso | -1 | 1/30 | 0.033 | 0.0279 | 0.9043950438499451 | 0.7338813122233027 |
| multiple_proxy | Gemma 4 E2B-it+Gemma 4 E4B-it | InternVL3.5-14B-HF | goldfish → monarch butterfly | -1 | 4/30 | 0.133 | 0.0802 | 0.9454782605171204 | 0.8397853935152674 |
| multiple_proxy | Gemma 4 E2B-it+Gemma 4 E4B-it | InternVL3.5-14B-HF | pineapple → acoustic guitar | -1 | 0/30 | 0.000 | 0.0500 | 0.8963858485221863 | 0.7134193890229025 |

## Result reasonableness checks

Validated 11040 frozen clean/adversarial PNG pairs; largest integer-pixel L∞ difference was 16/255 (limit 16/255). Checked 368 measured rows, 50-step settings, and explicit proxy-conditioned TASR counts. Strict parsing rejected 50 model outputs; these had no valid class code and are counted as target misses, with per-output records in `analysis/validation/target_output_diagnostics.csv`. The 17 ensemble clean-valid shortfalls are listed in the completion matrix. 17 ensemble batches had at least one non-class response, counted as target misses. 3 state(s) exceeded the 75 GiB reserved-memory target (maximum 77.29 GiB; no OOM). See the validation JSON files for details.

## Output map

`single_proxy/`, `layer_sweep/`, `multiple_proxy/`, and `ablation/` contain per-cell states and frozen PNGs. `analysis/embeddings/` stores reusable high-dimensional features. `analysis/pca/` and `analysis/tsne/` fit A/B references and both directions jointly. `analysis/variance/`, `analysis/representation_shift/`, `analysis/asymmetry/`, and `analysis/correlations/` contain the quantitative exports.

t-SNE is a qualitative view only; all reported geometric measures use the original embedding dimensions. Failed or unavailable cells are listed in `audits/cell_status.jsonl` and `analysis/embedding_extraction_failures.json`.
