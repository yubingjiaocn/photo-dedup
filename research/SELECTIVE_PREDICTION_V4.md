# Selective prediction and audit design (v4)

Scope is deliberately narrow: deciding which shadow decisions consume scarce
human review, and measuring the residual risk in the silent region. No source
adds deletion authority, and no Lin held-out data is read or tuned.

## Adopt

1. **Risk–coverage rather than ordinary accuracy.** Geifman and El-Yaniv,
   *Selective Classification for Deep Neural Networks* (NeurIPS 2017),
   arXiv:1705.08500, defines selective risk versus retained coverage and motivates
   ranking abstentions by estimated error risk. We adopt the curve and fixed
   review budgets, but use transparent cached-evidence rules because 20 bounded
   tune groups cannot support a reliable learned confidence model.
2. **Calibrate/validate confidence separately from fitting.** Guo et al.,
   *On Calibration of Modern Neural Networks* (ICML 2017), arXiv:1706.04599,
   shows accuracy and confidence calibration differ. We adopt the separation:
   the risk score is explicitly a ranking score, not a probability. Precision
   and silent error bounds remain unavailable until probability-sampled human
   audit labels exist.
3. **Active error discovery for the primary queue, probability sampling for
   estimation.** Kossen et al., *Active Testing: Sample-Efficient Model
   Evaluation* (ICML 2021), PMLR 139:5753–5763, uses active selection to find
   informative labels. We adopt expected-value ordering for discovery, but do
   not estimate population error from that biased queue. A separate deterministic
   stratified sample spans risk decile and changed/unchanged status and records
   inclusion probabilities/weights.
4. **Finite-sample binomial bounds.** NIST/SEMATECH e-Handbook, section
   7.2.4.1, *Confidence intervals for a binomial proportion*, documents exact
   binomial intervals; Brown, Cai and DasGupta, *Interval Estimation for a
   Binomial Proportion* (Statistical Science 2001), DOI 10.1214/ss/1009213286,
   compares Wilson/exact approaches. We report a one-sided 95% bound only after
   random diagnostic auditing. For zero errors in `n`, the exact upper bound is
   `1 - 0.05^(1/n)` (the rule-of-three is only an approximation). With multiple
   strata, report stratum estimates and a population-weighted estimate; do not
   treat an actively selected tune set as a simple random sample.

## Reject / defer

- **Learned confidence, temperature scaling, conformal risk control:** defer
  until there are enough event-level labels for train/calibration splits.
  Photo-level splits would leak near-duplicate evidence across groups.
- **Claiming <=10% review is safe from coverage alone:** reject. The 10% queue is
  an operational frontier point; safety requires labelled error-capture and a
  finite upper bound from the silent diagnostic sample.
- **Calling all uncertainty “diagnostic”:** reject. Diagnostic groups are a
  sampled audit workload, not a renamed exhaustive secondary queue.
- **Using tune-set errors as prevalence:** reject because the existing bounded
  visual tune set intentionally oversamples suspected failures.

## Operational audit

For each run, freeze the score, rank all groups, publish 3/5/10/20% frontiers,
then draw a deterministic sample from the silent population. The manifest keeps
anonymous dataset/group/member IDs, risk decile, group type, size bin,
changed/unchanged status, stratum population/sample counts, inclusion
probability, and analysis weight. Labels start `unlabelled`; only the four-button
human workflow may change them to `human_reviewed`. Re-run calibration only on
whole Willy events, never Lin held-out or photo-level folds.
