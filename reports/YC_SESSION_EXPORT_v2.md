# I ran an afternoon of neuroscience analysis, then spent two more hours trying to break it

**Author:** Abdul Muhsin Hameed ([@shnazzers](https://x.com/shnazzers), [abdulh.xyz](https://abdulh.xyz), [github.com/muhsinh/vbn-analysis](https://github.com/muhsinh/vbn-analysis))
**Tool:** Claude Code (Opus 4.7, 1M context). M3 Pro / 18 GB RAM.
**Dataset:** Allen Institute Visual Behavior Neuropixels, session 1055240613 — 734 quality-filtered neurons across 12 brain areas, 60 min active change-detection task + 60 min passive replay.

---

## What this session is

Three Claude subagents, each prompted to channel a composite of real working researchers in a sub-field, tore my own analysis apart.

- **Two of my headline findings didn't survive.** One was motor-confounded. One was a probe-registration artifact.
- **One finding got stronger after the critique than before it.** SCig active-dominance survives both arousal-matching and a miss-only sanity check. Full encoding R² doubled after fixing bugs the agents caught.
- **One novel result emerged that's not in the VBN literature at this scale.** Noise correlations reorganize by behavioral state across 12 brain areas — active-increase in visual cortex + SCig, active-decrease in midbrain reticular. I didn't look for this. An agent's critique of a different claim forced me to run it.

The thesis: **coding agents are most valuable as adversarial peer reviewers, not typists.** Spawn them in parallel with specific expert framings, take their critiques seriously enough to kill your own headline results, and you compress weeks of domain peer review into an afternoon.

---

## What happened in this session (compressed)

I opened with a single symptom: my notebook NB08 had consumed 60 GB of RAM and crashed. That unwound into two distinct bugs:

**The OOM** turned out to be a silent pipeline issue — 780 raw SuperAnimal keypoint columns feeding into a lag-expanded ridge regression design matrix (6,240 columns × 383k rows = ~20 GB per copy, amplified across CV folds). Fix: drop the keypoints at the NB07 merge step. 3.66 GB → 69 MB in RAM. 53× reduction.

**The deeper bug** only surfaced after the OOM fix, when the encoding model ran end-to-end and returned R² = **−1.67** on cross-validation — worse than predicting the mean, with a permutation test z-score of **−3.46** (real model is 3.46σ *below* the null distribution of random shuffles). A correctly-specified model cannot produce this under permutation by chance.

I didn't debug this alone. I fired a code-review subagent and a literature-survey subagent in parallel. Both came back with mergeable, specific findings. Three compounding root causes — no z-scoring, alpha grid saturating, and a silent einsum column-ordering bug in the raised-cosine basis expansion that was reading coefficients from the wrong columns. Three targeted fixes, applied in minutes.

Then I ran five deliverables end-to-end and, before shipping anything, **fired three more subagents** — a methodology critic (Pillow/Kriegeskorte lens), a VBN-insider (Siegle/Bennett/Piet lens), and a behavior-neural expert (McGinley/Stringer/Musall lens). Each got the full outputs and was asked to tear them apart.

Two survived critique. Two didn't. Details below.

---

## Five prompts that mattered

Not every prompt in a multi-hour session is load-bearing. These five did the work.

> **"peak ram was like 60 gb dawggggggg"**

12 words. Kicked off the methodology-deep-dive that found the 780-keypoint design-matrix blow-up, plus forced me to think about whether any head-fixed VBN literature motivates raw keypoint inclusion (answer: no). The casualness is the point — the leverage is in knowing which 12-word diagnosis to hand off, not in writing the prose around it.

> **"think about wht research says and if it makes sense do it"**

Delegation with a literature prior. The agent came back citing Stringer 2019, Syeda 2024, Musall 2019, Siegle 2021, and the head-fixed confound — then dropped the keypoints. No micro-management, no permission round-trips. The agent validates the prior against the actual field and executes.

> **"act as top researchers and leaders as this field in agents and think about these questions and any others, and critique whats done so far"**

Role-conditioning with specificity. I named Pillow/Park/Kriegeskorte/Simoncelli, Siegle/Bennett/Piet/Garrett, McGinley/Stringer/Musall/Churchland as the three composite personas. Generic "act as an expert" prompts are a yellow flag; named-person adversarial panels are the thing. This one call returned a convergence matrix where all three independently flagged **ZETA test to replace 4σ latency** as unanimous — and surfaced that my SCig "active engagement" finding was probably motor/lick contamination.

> **"whgats the core signal and do i have it? use multiple agents tyo think abt it and tell me"**

Meta-selection under uncertainty. Asking the tool to run the "is this thing real or am I fooling myself" check. Two independent investor-composite agents returned 8/10 and 8.5/10 with the same convergent critique about range vs depth. Using agents to disagree with you faster than any human advisor could.

> **"back to yc. is the yc claude md thing oyu made the right hting odr this position?"**

The meta-check on the output itself. Rare in the agent-user corpus. Agents tend to over-deliver on prompts as stated; asking "is what you made the right *kind* of thing" is the taste signal most users don't exercise. It's how v2 of this document exists.

---

## The critique that killed two of my findings

From the VBN-insider agent, verbatim:

> **SCig active-dominance is probably not task engagement — it's pre-lick motor preparation.** SCig is the deep layer of superior colliculus: orienting, saccades, licking. Piet's "midbrain task modulation" finding used strict no-lick windows. You haven't done that.
>
> **Fix**: (a) exclude change trials where `response_latency < 250ms`, (b) compute the same change amplification on *miss* trials only — if SCig still shows amplification on misses, it's sensory/attentional; if it collapses, it's motor.

I ran the miss-only test. **SCig amplification survived at 1.98× on trials with no lick at all.** The active-dominance signal is real — it's not motor prep. But the *same test* overturned a different finding:

> **DG ("change detection") collapsed on miss trials**: 1.49× → 0.81×. The hippocampal change signal is reward anticipation, not sensory. Same for CA1 (1.15× → 0.64×).

That's a novel result the critique would have caught in peer review anyway — except I caught it before submitting anything.

The convergence matrix from the three-agent critique panel:

| Issue | Methodology | Allen insider | Behav-neural | Call |
|---|---|---|---|---|
| 4σ latency → use ZETA (Montijn 2021) | ✓ | ✓ | ✓ | UNANIMOUS |
| Banded ridge (per-band α) | ✓ | — | ✓ | 2/3 high |
| Binary active/passive oversimplified | — | ✓ | ✓ | 2/3 high |
| SCig result may be motor-confounded | — | ✓ | — | Allen-authoritative |
| `pupil_vel` computed but never fed to model | — | — | ✓ | Behav-authoritative (real bug) |
| MGd (66 units) is misregistered probe | — | ✓ | — | Allen-authoritative |
| `gap_bins=20` inside raised-cosine kernel support | ✓ | — | — | real bug |

Applied the top-4 fixes. Re-ran. Documented what changed.

---

## Final scorecard after three iteration rounds

| Claim | Round 1 | After critique | Verdict |
|---|---|---|---|
| SCig active-dominance | +0.17 MI | 1.98× on miss trials, R² doubled after gap fix | **Stronger** |
| Change amp up visual hierarchy | VISal 1.3× → VISpm 2.3× | VISpm 4.5× on miss with adaptation-matched baseline | **Stronger** |
| Noise corr restructuring by state | — | +0.010 to +0.023 in visual cortex, −0.014 in MRN | **New, unpublished** |
| ProS as behavior-encoder (7.7% running R²) | — | Surfaced in fair Stringer test | **New, unpublished** |
| Visual cortex passive > active | −0.19 to −0.24 MI | −0.11 to −0.20 after arousal match | **Survives, weaker** |
| DG / CA1 "change detection" | 1.49× / 1.15× | 0.81× / 0.64× on miss | **Overturned — reward-driven** |
| MGd 5× change amp | reported | dropped | **Removed — probe artifact** |
| Stringer face-SVD replicates | Failed | Strawman — wrong target. Non-zero with pop-latent target. | **Resolved** |

Three claims got stronger. Two got killed. One got removed. One was a mis-specified test.

---

## Where this generalizes

The adversarial-agent-panel pattern is a 10× workflow improvement over the status quo in several markets:

- **Biotech / clinical trial reanalysis.** A $400/hr biostats consultant reviewing a pharma readout takes three weeks. Four parallel expert-framed agents with a full provenance log of what each critic flagged and how it was addressed take two hours at under $20 of compute. The incumbent can't adopt this workflow because their revenue model is the three weeks.
- **Quant research / factor validation.** Equity research shops running backtests have the same pathology — the person who builds the model is the person who reviews it. A three-agent panel (econometrician, portfolio manager, adversarial random-walk skeptic) catches overfits the author can't see.
- **Security red-teaming.** Most code review is done by the author's colleague. Three parallel agents (threat-model, exploit-chain, compliance) compress a week of peer review into an afternoon and produce an audit log that survives a regulator's review.

I am not claiming this session proves any of those markets. I am claiming the *pattern* is a repeatable primitive that scales past neuroscience. The same six steps that killed two of my headline findings apply directly.

---

## The actual point

Most people using coding agents right now are using them to write code faster. The real unlock is using them to **disagree with you faster** — to compress weeks of peer review into hours, and to keep you honest when the numbers come out looking good.

The code I shipped in this session is not the artifact. **The process of trying to kill my own findings before anyone else could is.**

That's what I'd bring to YC.

---

*Repo: [github.com/muhsinh/vbn-analysis](https://github.com/muhsinh/vbn-analysis). Full round-by-round findings: [reports/ROUND3_FINDINGS.md](./ROUND3_FINDINGS.md). Session iterations 1 & 2 addendum: [reports/NB08_FULL_ANALYSIS.md](./NB08_FULL_ANALYSIS.md). Twitter [@shnazzers](https://x.com/shnazzers).*
