# Counterfactual rule tests

This report tests behavioural movement on held-out postings, not ordinary accuracy. Each edit is deterministic and states are rebuilt with the 950-token posting budget; inserted gate sentences were checked for presence after truncation.

## Data and baseline

The run used **245** unique postings: 70 from `data/training/val.jsonl` and 175 from `data/experiments/export-v3/val2.jsonl`. No test split was read. The unedited verdict BASE agreement with the supplied gold labels was **186/245 (75.9%)**. Auxiliary agreement is available only for the 70 `val` rows: sponsorship 68/70, clearance 70/70, and min-years 68/70.

The baseline is a starting-point agreement check, not evidence that the model follows the policy. The counterfactual pass rates below are the primary results.

## Expectations and results

| Family | Frozen-policy expectation | n | Pass rate (95% Wilson interval) |
|---|---|---:|---:|
| years_up | 5+ required years -> `reject`, min-years `5+` | 84 | 89.3% (80.9%–94.3%) |
| years_down | 3+ required years replaced by 1–2 -> min-years `1–2`, not reject on years alone | 9 | 44.4% (18.9%–73.3%) |
| preferred | 5+ preferred/not required leaves verdict and min-years unchanged | 84 | 11.9% (6.6%–20.5%) |
| sponsorship | explicit no sponsorship -> sponsorship `true`, `reject` | 83 | 100.0% (95.6%–100.0%) |
| clearance | citizenship/clearance sentence -> clearance `true`, `reject` | 82 | 100.0% (95.5%–100.0%) |
| invariance | irrelevant edit leaves verdict unchanged | 239 | 98.3% (95.8%–99.3%) |

## Interpretation and typical failures

The clearest learned behaviour is the two explicit hard-gate families: sponsorship and clearance both passed 83/83 and 82/82. Years-up movement passed 89.3%, but the preferred-years control passed only 11.9%; the model is therefore sensitive to years language but does not reliably distinguish preferred from required. Years-down is a small, conservative proxy sample (9 cases) and is correspondingly uncertain. The retained invariance controls were stable at 98.3%, with 6 controls excluded when the irrelevant edit did not survive state construction.

A pass means the typed answer met the family-specific expectation; it does not prove causal understanding. A failure can be a missed gate, a wrong score bucket, or an over-sensitive verdict boundary. For years-down, selection is intentionally conservative: base prediction is reject, min-years is 3+, and the model has no predicted sponsorship/clearance gate; this is a proxy for ‘years-only’, not an independent proof of the human reason.

### years_up
75 of 84 cases passed.
Typical redacted failure excerpts:
- [Company] after `Requires 5+ years of professional software experience.`: missing required years movement

### years_down
4 of 9 cases passed.
Typical redacted failure excerpts:
- [Company] after `Requires 1-2 years of relevant experience.`: years-down response still rejects or has wrong bucket

### preferred
10 of 84 cases passed.
Typical redacted failure excerpts:
- [Company] after `5+ years of relevant experience preferred, but not required.`: preferred years changed a typed answer

### sponsorship
83 of 83 cases passed.
No failure excerpt was recorded.

### clearance
82 of 82 cases passed.
No failure excerpt was recorded.

### invariance
235 of 239 cases passed. 6 edited states were omitted because the controlled text did not survive state construction.
Typical redacted failure excerpts:
- [Company] benefits edit: verdict flipped on irrelevant edit
- [Company] typo edit: verdict flipped on irrelevant edit
- [Company] company edit: verdict flipped on irrelevant edit

## Review notes: what the failures actually are

- **"Preferred" years are read as a requirement.** Adding "5+ years of relevant experience preferred, but not required." moved 71 of 84 strong/stretch postings to `reject`. In 47 of them the min-years answer also jumped to `5+`. The model learned "5+ years" as a reject trigger and largely ignores "preferred". This is the clearest shortcut found. A likely cause is the training data, which rarely contrasts required with preferred years.
- **Years-up failures are bucket errors, not verdict errors.** All 9 failures did move to `reject`; only the min-years answer stayed at `1–2`.
- **Years-down has n=9**, too few to conclude anything; the interval is 19%–73%.
- **Invariance:** 4 edit types (company name, EEO boilerplate, a typo, benefits order). 4 of 239 flipped, all from `stretch`, the class nearest the decision boundary.
- Sponsorship and clearance gates are perfect (165/165): explicit, consistently worded gates are learned well.

Figure: `docs/figures/rule-tests.png` (`scripts/make_rule_tests_figure.py`).

## Limits

These are paired model self-tests on held-out text, not new human labels. The exported Q8_0 model can react to wording, truncation, and correlated posting artifacts. The policy allows relevant alternative experience for a 3+ minimum; the years-up edit deliberately does not add such an allowance. Controls test only verdict invariance, so stable answers do not establish invariance of every probability.
