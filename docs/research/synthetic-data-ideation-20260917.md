# Ideation: synthetic training data for Brandy's high register

**Status: PROPOSALS. Nothing here is a finding.** Labels used throughout:
`[idea]` `[assumption]` `[prediction]` `[evidence]` `[decision]`.

Session id: synthdata-01 · 2026-09-17 · decision owner: user (GPU spend)
Facilitator/generator: Claude (single-agent; see "Perspectives missing")

## 1. Scope

**Focal question.** How can we manufacture training data from Brandy's existing
corpus such that the model stops degrading above ~500 Hz?

**Why this question and not "get more data".** `[evidence, this project]`
Every local source has been measured for >700 Hz content and identity; none is
her singing up there. Booktalk was tested and is not her (0.786 vs her
0.95-0.97). So the register gap cannot be closed by ingest.

**What is already known — this bounds the search.** `[evidence, measured this session]`

| fact | number |
|---|---|
| corpus | 270 plain + 64 WORLD +12 st + 64 WORLD +16 st |
| augmented data is CLEAN | inharmonic 0.343 (+12) / 0.365 (+16) vs real voice 0.329 |
| model degrades with pitch | dropout 1.9 % at 300-400 Hz -> 16.6 % at 700-1000 Hz |
| buzz degrades with pitch | inharmonic excess +7 % -> +41 % over the same range |
| loss reweighting helped | raw dropout 63.4 -> 36.1 %, inharmonic +70 -> +52 % |
| identity metric | mel-stat centroid, floor 0.88, currently 0.9998 |
| cost of one fine-tune | ~20 min, 150 epochs, RTX 5090 |
| a single render CANNOT measure dropout | 3 renders of one model spanned 6.1-50 % |

**The central puzzle.** `[evidence]` The augmented high-register data measures
nearly as clean as a real voice, and the model still fails up there. Loss
reweighting recovered roughly half the deficit, which says the data is being
under-learned rather than being wrong. But half remained. So either the data is
still under-weighted, or it is *realistic in the ways we measured and
unrealistic in a way we did not*.

**Out of scope.** Ingesting new recordings of other singers as Brandy;
anything that changes who the voice sounds like (identity floor is a gate, not
a criterion); inference-time-only tricks are IN scope but labelled separately
since they are not "training data".

**Prohibited.** Nothing here touches personal data, human subjects, or
dual-use. The one real risk is **voice likeness**: this is a consented,
authorized voice-model project; synthetic data derived from her own recordings
stays within that consent, and no other artist's identity may be introduced
into her model. That is an ethics gate, checked in §7.

## 2. Perspectives represented and missing

Represented: signal processing (WORLD/HNM), ML training dynamics, the
deployment constraints of this specific fork.

**Missing, and it matters:** a singing teacher or phonetician. Every idea below
rests on assumptions about how a real voice changes with register, and I have
no expert in the room to challenge them. Flagged again in §7.

## 3. Independent generation (round 1, before any literature search)

Generated before searching, deliberately, so the search does not anchor them.

**Cluster A — make the shifted data physically more like a real high note**

- `A1` **Formant-corrected pitch shift.** WORLD preserves the spectral
  envelope, so a 300 Hz clip shifted to 600 Hz keeps 300 Hz formants.
  `[assumption]` A real singer's formants move with register (F1 rising to
  track f0 in belt). `[prediction]` Training on formant-corrected shifts
  reduces high-f0 inharmonicity more than plain WORLD shifts.
  `[why it could matter most]` At inference the SOURCE is a real singer at
  700 Hz with 700 Hz formants; the model was trained on Brandy-at-700 Hz with
  300 Hz formants. That is a train/test mismatch exactly where it fails.
- `A2` **Aperiodicity transplant.** Keep her envelope, take band-aperiodicity
  from a real singer's high notes. `[assumption]` Aperiodicity at high f0 is
  more speaker-generic than the envelope is.
- `A3` **Harmonic-amplitude extrapolation.** Rather than shifting a whole clip,
  resynthesize a high note from her measured harmonic amplitude/phase rules,
  extrapolated in f0.
- `A4` **Smaller shifts, denser coverage.** +2..+6 st from her *upper* existing
  material instead of +12/+16 from her modal range. `[assumption]` Shift
  artifacts and formant error both scale with shift size, so many small shifts
  beat few large ones.

**Cluster B — change what the training set emphasises, not what is in it**

- `B1` **f0-histogram balancing.** Resample so the f0 histogram is flat rather
  than peaked at her modal range. Data-level sibling of the loss weighting
  that already worked.
- `B2` **Curriculum over register.** Train modal first, then progressively
  admit higher shifts.
- `B3` **Segment-level rather than file-level augmentation.** Augment only the
  frames near the failure boundary.

**Cluster C — generate new material with another model**

- `C1` **Self-training / pseudo-labelling.** Convert a high-register singer to
  Brandy with the CURRENT model, add to training data.
  `[assumption]` model output is good enough to teach itself.
  `[predicted failure]` This is a confirmation loop: it teaches the model to
  reproduce its own dropouts and buzz. Listed because it is the obvious idea
  and should be explicitly killed rather than silently skipped.
- `C2` **External SVS/diffusion synthesis** of high-register Brandy from her
  embedding, used as training data.
- `C3` **Voice-conversion round trip** through a third voice with real high
  notes, mapping back to her timbre.

**Cluster D — sidestep the extrapolation at inference (not training data)**

- `D1` **Convert-low-then-shift-up.** Shift the SOURCE down into her comfortable
  range, convert, shift the OUTPUT back up with WORLD. The model never
  extrapolates. `[prediction]` Removes high-f0 dropout almost entirely;
  `[predicted cost]` output inherits WORLD artifacts and possibly wrong
  formants — the mirror image of A1's problem.
- `D2` **Register-split rendering.** Use D1 only on frames above a threshold,
  crossfade with the direct conversion below it.

**Cluster E — generic augmentation (listed for completeness, low prior)**

- `E1` VTLP / vocal-tract-length perturbation. `[gate risk]` directly perturbs
  identity, which is the one thing we cannot trade.
- `E2` noise/reverb/EQ robustness augmentation — does not address register.
- `E3` latent-space mixup in ContentVec features.

## 4. What became LESS obvious while writing this

`C1` looked attractive and is probably actively harmful. `E1` is standard
practice in ASR and is close to disqualified here because identity is a gate.
And `A4`/`B1` are the cheapest things on the list yet were not the first ideas
to surface — large shifts are more interesting to think about than small ones,
which is exactly the kind of preference that should not drive the decision.

---

## 5. Evidence check (searches run 2026-09-16)

### 5.0 What was searched, and what was actually read

**WebSearch queries, verbatim, all run 2026-09-16:**

1. `SingAug Data Augmentation Singing Voice Synthesis Cycle-consistent Training Strategy arXiv Guo Shi Qian Watanabe Jin`
2. `singing voice conversion target speaker training data lacks pitch range extension out-of-range f0 extrapolation`
3. `"vocal range" extension singing voice synthesis model generate notes outside training corpus pitch range generalization`
4. `Joliveau Smith Wolfe 2004 tuning of vocal tract resonance by sopranos R1 tracks f0 above F1`
5. `Bourne Garnier belting resonance tuning R1 f0 acoustic analysis belt musical theatre female JASA first formant`
6. `formant correction pitch shifting data augmentation singing voice conversion formant shift spectral envelope warping training data`

**Sources fetched directly (not via search) and the depth actually read.** Anything
marked *abstract only* is not paraphrased beyond its abstract.

| # | Source | Read |
|---|---|---|
| S1 | SPA-SVC, Bai/Wang/Gao/Li, Interspeech 2024, arXiv:2406.05692 — `arxiv.org/html/2406.05692v1` (+ ar5iv mirror, identical) | **full text** |
| S2 | PDAugment, Zhang et al., arXiv:2109.07940 — `ar5iv.labs.arxiv.org/html/2109.07940` | **full text** |
| S3 | SingAug (= guo22e), Guo/Shi/Qian/Watanabe/Jin, Interspeech 2022 pp. 4272–4276, doi 10.21437/Interspeech.2022-978, arXiv:2203.17001v2 — ISCA abstract page + `arxiv.org/html/2203.17001v2` | **full text** |
| S4 | Zhou et al., "Enhancing the Vocal Range of Single-Speaker SVS with Melody-Unsupervised Pre-training", arXiv:2309.00284v1 | **full text** |
| S5 | WeSinger, Zhang/Zheng/Li/Lu, arXiv:2203.10750 — ar5iv | **full text** |
| S6 | Joliveau, Smith & Wolfe, "Vocal tract resonances in singing: The soprano voice", JASA 116(4) Pt.1 (2004) 2434–2439, doi 10.1121/1.1791717 — author reprint PDF at `phys.unsw.edu.au/~jw/reprints/Joliveauetal.pdf` | **full text** |
| S7 | Henrich, Smith & Wolfe, "Vocal tract resonances in singing: Strategies used by sopranos, altos, tenors, and baritones", JASA 129(2) (2011) 1024, doi 10.1121/1.3518766 — PubMed 21361458 | *abstract only* |
| S8 | Bourne & Garnier, "Physiological and acoustic characteristics of the female Music Theater voice", JASA 131(2) (2012) 1586, doi 10.1121/1.3675010 — PubMed 22352528 | *abstract only* |
| S9 | Liu et al., "Self-supervised restoration of singing voice degraded by pitch shifting using shallow diffusion", arXiv:2601.10345 (Jan 2026) | *abstract only* |

The `isca-archive` PDF for SPA-SVC was not used; the arXiv HTML was, as anticipated.

### 5.1 SPA-SVC — the actual mechanism (S1, full text)

This is the paper the project most needs to read correctly, because **it does not
create pitch-shifted training files at all.** `[evidence, S1 §2.1]`

Training, per step:

1. Extract `SSL_o` (ContentVec, 16 kHz / hop 320), `Vol_o`, `F0_o` (RMVPE), speaker ID
   from the real audio `wav_o`.
2. Shift **the f0 conditioning sequence only**: `F0_s = F0_o × 2^(key/12)`, with
   `key` a **random integer in [6, 18] semitones**. Envelope, aperiodicity and
   content features are untouched because no signal is resynthesised — there is
   nothing to preserve or corrupt.
3. Pass 1: `SSL_o + F0_s + Vol_o + spk` → DDSP model → `wav_s`, "a pitch-shifted
   singing voice with added breathiness".
4. Re-extract `SSL_s` from `wav_s` (i.e. ContentVec applied to the model's own
   high-pitch render).
5. Pass 2: `SSL_s` + **the original** `F0_o, Vol_o, spk` → the **same** DDSP model
   → `wav_c`, "with the original pitch restored".
6. `L_cyc = 1 − SSIM(mel(wav_o), mel(wav_c))` — SSIM is computed on
   **mel-spectrograms**, and it **is** the cycle-consistency loss, not an extra
   reconstruction term.
7. `L_total = L_cyc + L_diff`. `L_diff` is the ordinary diffusion loss on the
   cycle-reconstructed hidden features.

The supervision target is therefore **always real audio at its real pitch**. Model
output appears only as an intermediate. `[evidence]`

Four things that bound how much this result is worth:

- **Gradient flow is not stated.** `[verified absent]` The full text was grepped for
  `gradient|detach|backprop|frozen|freeze|stop.grad|no_grad|differentiab`; the only
  match is a reference title (DDSP, "Differentiable digital signal processing"). The
  paper contains **no statement** on whether `SSL_s` is detached. If gradients flow
  through pass 1, the mechanism is the stated one ("optimise the DDSP model to
  generate high-pitched singing voices less likely to contain hoarseness"). If
  `SSL_s` is detached, the mechanism is instead "make the decoder robust to corrupted
  content features at normal pitch", which is a different claim with a different
  implication for F1. `[assumption]` — do not report either reading as evidence.
- **Both arms already use small-shift augmentation.** "for fair comparison, both our
  model and the DDSP-SVC-Diff model employed minor fundamental frequency random
  perturbations within the semitones of (−5, 5) range as the basis for data
  augmentation." `[evidence, S1 §3.2]` The big-shift cycle is measured *on top of*
  ±5 st perturbation, not against no augmentation. Ref [13] in S1 is that
  "minor-range pitch enhancement in other models" — the A4 prior art.
- **No objective metrics at all. MOS only, 13 listeners**, 12 test segments plus 48
  original/shifted segments. `[evidence, S1 §3.3]` Given this project's own finding
  that three renders of one model spanned 6.1–50 % dropout, a 13-listener MOS is
  weak evidence for effect size.
- **The gain tracked the size of the range gap, and the small-gap target got worse.**
  Table 1 (95 % CI):

  | | MOS-A (alto tgt) | MOS-B (bass tgt) | MOS-O | MOS-S | MOS |
  |---|---|---|---|---|---|
  | GT | – | – | – | – | 4.30 ± 0.12 |
  | DDSP-SVC-Diff | 3.95 ± 0.09 | 3.27 ± 0.07 | 3.70 ± 0.10 | 3.52 ± 0.08 | 3.61 ± 0.06 |
  | SPA-SVC-M (cycle + MSE) | 3.82 ± 0.09 | **3.60 ± 0.07** | 3.77 ± 0.09 | 3.65 ± 0.07 | 3.71 ± 0.06 |
  | SPA-SVC (cycle + SSIM) | 3.96 ± 0.09 | 3.56 ± 0.07 | 3.82 ± 0.09 | 3.69 ± 0.08 | 3.76 ± 0.06 |

  MOS-B (bass singer pushed high — the large-gap case) gains +0.29/+0.33, well
  outside the CI. MOS-A (alto target, small gap) gains **nothing** (3.95 → 3.96) and
  the cycle *without* SSIM actively lost 0.13. And on the metric that matters most
  here, MOS-B, **MSE beat SSIM** (3.60 vs 3.56). `[evidence]` So: the cycle is the
  active ingredient, SSIM's contribution is +0.05 overall and negative on the
  cross-domain-large-gap metric (3.60 vs 3.56, **within the ±0.07 CI** — read as "no
  SSIM advantage there", not as MSE winning), and the whole benefit is conditional on
  a large range gap.

  `[relevance]` Brandy is structurally the **Bass-3** case, not the Alto-1 case: her
  p99 is 407 Hz and the source reaches 1100 Hz, a gap of roughly 1.5 octaves.

Base system: DDSP-SVC-Diff (DDSP CombSubFast + WaveNet diffusion decoder) +
**NSF-HiFiGAN** vocoder, ContentVec SSL, M4Singer (Mandarin), 554 training segments
for Alto-1 / 967 for Bass-3, 40k steps at batch 64. `[evidence, S1 §3.1–3.2]`
`[assumption]` Transfer to a so-vits-svc-fork VITS model with a posterior encoder is
untested by anyone; running the generator twice per step is a non-trivial port.

### 5.2 What the augmentation literature says helps — and what it says does not

**SingAug / guo22e (S3, full text).** Pitch augmentation (PA) is implemented exactly
as this project does it: "By multiplying or dividing ¹²√2 to F0 sequence and
**keeping the SP and AP unchanged**", via WORLD. `[evidence, S3 §2.2]` So the
"WORLD preserves formants" property is the published norm, not a deviation.

Two tables decide several of the ideas below.

*Table 2 — how big should the shift be (Opencpop; MCD ↓, ST_ACC ↑, VUV_ERR ↓):*

| PA variant | MCD | ST_ACC | VUV_ERR |
|---|---|---|---|
| P-adaptive (shift samples toward the dataset mean pitch) | 8.26 | 73.29 | 6.77 |
| **P1 = random {−1, 0, +1} st** | **7.85** | **74.37** | **5.68** |
| P2 = random {−2 … +2} st | 8.08 | 73.92 | 6.06 |

Verbatim: "the variation range of PA is not proportional to the synthetic quality.
Too much variation in PA may reduce the effectiveness of the acoustic model."
`[evidence, S3 §3.4A]` **±1 semitone beat ±2 semitones on all three metrics**, and
the *targeted* variant — move samples toward a chosen pitch region — was the **worst
of the three**.

*Table 5 — does PA help at all (Opencpop):*

| | MCD | ST_ACC | VUV_ERR |
|---|---|---|---|
| CC2 (cycle-consistent only) | 8.06 | 73.67 | 6.63 |
| CC2 + PA | 8.11 | 73.46 | 6.44 |
| CC2 + MA (mix-up) | 8.04 | 74.28 | 6.20 |
| **CC2 + PA + MA** | **7.76** | **74.30** | **5.71** |

`[evidence, S3 §3.4C]` **Pitch augmentation on its own degraded MCD and semitone
accuracy.** It paid off only in combination with mix-up. Mix-up alone improved all
three.

And on Ofuton: PA/MA/PA&MA&CC improved everything "except on the VUV_ERR metric.
This may be due to the artifacts introduced from the semitone shift on the training
data when PA is applied." `[evidence, S3 §3.3]` **Pitch augmentation specifically
damages voiced/unvoiced behaviour** — which this project already has as a standing
defect.

SingAug's stated motive for its cycle-consistent strategy is the same suspicion this
doc calls its "central puzzle": "For PA, the noise may come from the WORLD vocoder
when modifying target singing signals, as its quality still has a gap with the real
singing voice ... These artifacts and noises may disrupt the training of the SVS
acoustic model." `[evidence, S3 §2.3]` Their CC is **not** SPA-SVC's cycle: it is an
auxiliary predictor `g(Y)` mapping acoustic features back to phoneme and pitch
sequences under cross-entropy, weighted `w_svs=0.85, w_si=0.1, w_pd=0.05` (best of
three weightings).

**WeSinger (S5, full text)** — the only located paper that tried pitch shifting
*specifically to extend vocal range*, and abandoned it. Verbatim: "the overall pitch
of each song is raised by one semitone and lowered by one semitone in turn ...
However, we find that doing so could lead to somewhat perceptible changes in timbre.
Based on the existing VS method, we found no significant benefit from adopting the
additional pitch-shifting method. Therefore, in this paper, we only present the
performance of adopting the VS method." `[evidence, S5 §2]` Caveat: this is an
author's discard note with **no ablation table**, so it is weak evidence — but it is
weak evidence pointing the same way as SingAug Table 5.

**PDAugment (S2, full text)** — honest reframe: the task is **automatic lyrics
transcription**, not SVS or SVC. Speech is warped toward the singing domain to train
a *recogniser*; the metric is WER. It says nothing directly about perceptual quality
of a generated voice. What it does contribute:

- Pitch adjustment is WORLD, F0-only, "synthesize speech with adjusted F0 accompanied
  with original aperiodicity and spectral envelope". `[evidence, S2 §3.3]`
- **It caps the shift.** "Considering that the quality of synthesized speech will drop
  sharply when the range of adjustment is too large, we need to keep it within a
  reasonable threshold ... which is 5 in our experiment" — beyond 5 semitones they
  transpose the whole note sequence instead of shifting further. `[evidence, S2 §3.3]`
- Targeted beats random: score-guided syllable-level adjustment 9.8 % test WER vs
  17.6 % for random ±6 st / random 0.5–1.2× duration vs 20.5 % for unmodified speech.
  Ablation: removing the pitch adjuster costs 3.6 points WER, removing the duration
  adjuster 4.0 points. `[evidence, S2 §5.1–5.2, Tables 2 and 4]`

### 5.3 Does anyone address a corpus that genuinely lacks the required range?

**Closest located: S4 (arXiv:2309.00284), and it is not quite this problem.**
It states the premise exactly — "The single-speaker SVS system also fails at pitch
values that are associated with limited training samples, even if they are within
the singer's vocal range" — and solves it with **melody-unsupervised multi-singer
pre-training, then single-singer fine-tune**, not with augmentation. Ablation: removing
pre-training costs **−0.857 CMOS**, by far the largest of its three contributions
(bi-directional flow −0.335, differentiable duration regulator −0.143). Its
range-specific evidence is F0 MAE / F0 correlation recomputed on test segments
**below F3 or above D5**, where the gap over the baseline widens. `[evidence, S4 §4.1–4.3]`

`[honest limitation]` D5 ≈ 587 Hz — *below* this project's failure band. Their case
study is a **low** pitch with few samples. Their corpus (Opencpop, 5.2 h) has
*sparse* coverage in the extremes, not *zero*. Brandy's corpus is 0.00 % above 700 Hz.

**Everything else located solves the problem on the source side, not the target
side.** SPA-SVC's own inference recipe transposes the source key per pair based on
the range difference (keys of 4 or 12 st depending on the source/target pairing,
S1 §3.2); the general SVC practice surfaced by query 2 is linear F0 scaling by the
ratio of mean F0s. That is D1, not range extension.

**Finding: no direct evidence located, in a bounded search, for genuine range
EXTENSION of a target singer whose corpus contains zero frames in the required band.**
`[evidence: absence]` This is a negative result from six WebSearch queries and nine
fetched sources on 2026-09-16, not a survey. **It is not a novelty claim** — the
adjacent literature is large, much of it is behind PDFs this search did not open, and
"nobody published it" and "nobody found it worth publishing" are indistinguishable
from here.

### 5.4 How formants actually behave as a singer ascends — this decides A1

`[evidence, S6, full text]` Trained sopranos, resonances measured directly by
broadband excitation at the mouth (so independently of the harmonics):

- While `f0 < R1_speech`, **R1 and R2 stay put** at their speech values.
- Once `f0` exceeds the speech value of R1, **R1 rises to track f0**, so that
  `R1 ≈ f0`. "The tuning line R1 = f0 lay within the standard deviations of the data
  in the approximate range **600 to 1000 Hz**."
- Worked example: one singer's /u/ had R1 ≈ 420 Hz at C4 (261 Hz); at C6 (1046 Hz)
  "R1 and f0 now coincide to within ~20 Hz", and R1–R4 had all increased.
- **R2 rises too, but "proportionally smaller than that measured for R1"** — a side
  effect of the jaw/mouth opening used to raise R1, not a tuning target.
- **R3 "increased slightly but systematically, across the whole pitch range."**
- The tracking **saturates**: for rounded vowels (/ɔ/, /u/) the data fall below the
  tuning line near 1 kHz — "with the lips rounded, it is uncomfortable or
  anatomically impossible to raise R1 to 1 kHz."
- Consequence: "vowels move, converge, and overlap their positions on the vocal plane
  (R2,R1) to an extent that implies loss of intelligibility."

`[evidence, S7, abstract only]` The strategy is **voice-type and singer dependent**:
sopranos use R1:f0 (and often R2:2f0 simultaneously); altos use R1:f0 for
low-R1 vowels but can switch to R1:2f0 lower in the range; tenors and baritones use
R1:2f0 and R1:3f0. "there can be considerable differences in the resonance strategies
used by individual singers".

`[evidence, S8, abstract only]` **Belt is a different tuning again.** In chesty and
twangy belt, female music-theatre singers "systematically tuned the frequency of their
first vocal tract resonance (R1) to the **second harmonic (2f0)** up to C5" (≈523 Hz),
measured over F#4–D5 (~370–600 Hz). R2 was **not** tuned in any quality, though it sat
higher in belt than legit. Also: "the energy of the spectrum above 1 kHz" was
significantly greater in chesty belt than legit.

**What this does and does not do for A1.**

- It **supports A1's `[assumption]`** decisively, and the band matches: the onset
  region 600–1000 Hz sits inside the project's 700–1100 Hz failure band. Brandy's
  corpus (median 291 Hz, p99 407 Hz) contains **essentially no examples of resonance
  tuning at all** — every frame is in the regime where R1 is held at its speech value.
  WORLD +12/+16 st moves f0 to roughly 582–1026 Hz while **leaving R1 exactly where it
  was**, producing sustained `f0 > R1` configurations that a real singer retunes out of.
  That is a concrete, named, previously-unmeasured way in which the augmented corpus
  is unrealistic — exactly the shape of the doc's "central puzzle". `[prediction]`
- It **challenges A1's proposed implementation.** "Scaling the spectral envelope" by
  the pitch ratio is the wrong transform. The real warp is: R1 → n·f0 (n = 1 for
  soprano, 2 for belt/alto-low, 2–3 for male), R2 up only slightly and incidentally,
  R3 up very slightly, and the whole thing **saturates** near 1 kHz and is
  vowel-dependent. A uniform ×2 warp would drag R2, R3 and the singer's-formant region
  — where identity lives — far past anything a real voice does.
- `[gate risk, unresolved]` **A1 and E1 are the same operation** — frequency-warping
  the spectral envelope — differing only in intent. If E1 carries an identity-gate
  risk, A1 inherits it, and the formant literature does nothing to remove it. WeSinger's
  report of perceptible timbre change from a **±1 semitone** shift (which does *not*
  warp the envelope) suggests the identity margin here is thinner than assumed.
- `[assumption, unresolvable from literature]` "Match the source singer's formants" is
  not a single target: the inference source's tuning depends on her technique
  (classical R1:f0 vs belt R1:2f0 vs pop somewhere between), and belt tuning is
  documented only up to C5.

### 5.5 Per-idea status

Statuses: **not-checked** (no search run, or none that could have found it) ·
**search-incomplete** (searched, nothing located, absence is weak) ·
**support-located** · **challenge-located** · **mixed**.

| Idea | Status | One-line reason |
|---|---|---|
| **A1** formant-corrected shift | **mixed** | Assumption strongly supported (S6: R1→f0 in 600–1000 Hz, the exact failure band); the *implementation* (uniform envelope scaling) is contradicted — the real warp is non-uniform, saturating, vowel- and style-dependent (S6/S7/S8). No SVC paper located that does formant-corrected pitch-shift augmentation. Inherits E1's identity gate. |
| **A2** aperiodicity transplant | **search-incomplete** | Nothing located. Note S2/S3 both explicitly keep AP unchanged; no located work transplants it across singers. |
| **A3** harmonic-amplitude extrapolation | **not-checked** | No query in this round would have found it; no evidence either way. |
| **A4** smaller shifts, denser coverage | **mixed** | Strong support for *small*: S3 Table 2 (±1 st beats ±2 st on all metrics; "too much variation … may reduce the effectiveness"), S2 caps at 5 st ("quality … will drop sharply"), S1's baseline is ±5 st, S9 abstract ("artifacts … more severe at larger transposition jumps"). But no support for it *helping*: S3 Table 5 shows PA alone degrades MCD/ST_ACC, S5 found "no significant benefit". Small is safer, not proven better. |
| **B1** f0-histogram balancing | **mixed** (weak challenge) | The nearest published analogue, S3's "P-adaptive" (shift samples toward the dataset mean pitch), was the **worst** of three PA variants on all three objective metrics. Flattening ≠ centring, so this is a caution, not a refutation. No direct work located. |
| **B2** curriculum over register | **search-incomplete** | Nothing located in this search. |
| **B3** segment-level augmentation | **support-located** (indirect) | S2's whole result is that sub-utterance, *targeted* adjustment massively beats whole-file random (9.8 % vs 17.6 % WER). Transfer caveat: that is ALT, and "targeted by music score" ≠ "targeted by failure boundary". |
| **C1** self-training on own output | **challenge-located** | Confirmed as written. S1 is the closest published system and it deliberately **never uses model output as a target** — the high-pitch render is an intermediate; the loss target is always the real original mel. The doc's predicted confirmation loop is the reason, and S1 shows the repair (→ F1). |
| **C2** external SVS/diffusion synthesis | **search-incomplete** | Nothing located. Identity/consent gate untouched by any evidence found. |
| **C3** round trip through a third voice | **search-incomplete** | S1 and S3 are both "cycle consistency" but neither routes through another singer's real high notes; nothing located that does. |
| **D1** convert-low-then-shift-up | **support-located** | This is standard published inference practice, not a workaround: S1 §3.2 transposes the source key per pair by the range difference (4 or 12 st); query 2 surfaced mean-F0-ratio source scaling as the common SVC remedy. Its predicted cost now has a named published fix (S9, abstract only: self-supervised restoration of pitch-shift-degraded singing via shallow diffusion). |
| **D2** register-split rendering | **not-checked** | No located work on threshold-gated hybrid rendering. |
| **E1** VTLP | **challenge-located** (identity) | No SVC-specific VTLP result located, but S5 reports perceptible timbre change from a shift that does *not* even touch the envelope — VTLP touches it by design. Also operationally identical to A1. |
| **E2** noise/reverb/EQ | **not-checked** | Out of scope for register by the doc's own reasoning; not searched. |
| **E3** latent mixup in ContentVec | **support-located** (indirect) | S3's mix-up (MA) on expanded hidden states was the **best-supported single augmentation in the closest paper**: CC2+MA improved all three metrics over CC2, and PA only paid off when combined with MA. Caveat: SVS hidden states from a score, not ContentVec features in an SVC model. **This was the doc's lowest-prior idea and has the best evidence of any of the fifteen.** |

---

## 6. Round 2 ideation (reopened after the evidence check)

New ideas the literature suggests, not present in round 1.

- `F1` **In-graph cycle pitch augmentation (port SPA-SVC's mechanism, not its files).**
  `[idea]` At train time, shift only the f0 conditioning by a random +6…+18 st,
  render, re-extract ContentVec from that render, run a second pass with the
  **original** f0, and take the loss against the **real** original mel. No offline
  corpus, no WORLD, no envelope question — because nothing is resynthesised into a
  file. `[why it matters]` It attacks the register gap without ever handing the model
  a physically wrong high note, and it sidesteps every A1/A4 formant problem by
  construction. `[assumption]` Transfers to a VITS fork with a posterior encoder —
  untested by anyone. `[predicted cost]` Two generator passes per step (~2× step
  time), plus the open question of whether the ContentVec path needs to be
  differentiable (S1 is silent on detachment — see §5.1).
- `F2` **SSIM mel loss as a standalone one-line trial.** `[idea]` Cheap to try, but the
  evidence says small: +0.05 MOS overall in S1, and **negative** on MOS-B, the
  large-range-gap metric that matches this project. `[prediction]` Below the noise
  floor of a 3-render dropout measurement. Listed so it is not mistaken for the
  paper's main result — the cycle is.
- `F3` **Interrogate the pre-trained base's range coverage before training anything.**
  `[idea]` S4's largest single ablation is pre-training (−0.857 CMOS), and both S4 and
  S5 reach range/robustness via multi-singer pre-train → single-singer fine-tune —
  which is *already the fork's architecture*. `[measurement, not a training run]`
  Measure the f0 histogram of the base checkpoint's training corpus above 700 Hz. If
  the base is itself starved up there, no amount of target-side augmentation can
  supply what the decoder never learned, and swapping the base is a cheaper lever than
  any idea in cluster A.
- `F4` **Auxiliary predictor head on the augmented data (SingAug's CC).** `[idea]` Add
  a light head predicting f0 (and/or content) from the generated mel, cross-entropy,
  weighted low (S3's best: `w_svs=0.85, w_si=0.1, w_pd=0.05`). `[why]` Its stated
  purpose is precisely to stop WORLD's artifacts from disrupting training — the doc's
  central puzzle, named by the authors. `[evidence]` It is the component that made PA
  profitable at all in S3 Table 5.
- `F5` **Restore the shifted corpus before training on it.** `[idea]` S9 (abstract only)
  trains a shallow mel-space diffusion model to undo pitch-shift artifacts, building
  pairs self-supervised by shifting and un-shifting. The same pairs can be built here
  for free from Brandy's own corpus. `[two uses]` clean the +12/+16 training files
  before they enter training, **or** clean D1's output after inference — the same model
  serves both. `[prediction]` Higher value on the D1 path, because there it removes
  D1's only named cost.
- `F6` **Check R1-vs-f0 on the augmented corpus.** `[idea, and the cheapest thing
  here]` The doc's puzzle is "realistic in the ways we measured, unrealistic in a way
  we did not". §5.4 names a specific unmeasured axis with a published ground truth:
  in real female singing, `R1 ≈ f0` once f0 exceeds R1_speech, with the tuning line
  holding across 600–1000 Hz.
  **Do not try to measure R1 from real high-register audio** — S6 used broadband
  excitation at the mouth precisely because "it is difficult to estimate the tract
  resonances from the voice signal, simply because the spacing between harmonics is
  too great". Three tractable substitutes:
  1. *Augmented side, exact:* read R1 straight off the WORLD/CheapTrick spectral
     envelope already computed for each shifted file — it is constructed, not
     estimated, so there is no ambiguity.
  2. *Real side, no measurement:* use S6 Fig. 3's published per-vowel averages as the
     ground-truth R1(f0) curve.
  3. *Audio-observable on both sides:* **H1 prominence.** If R1 is frozen at ~350–450 Hz
     while f0 sits at 733 Hz, the fundamental lands on the envelope's skirt and is weak;
     a real R1≈f0 voice has a boosted H1 — that is what the tuning buys. Compare
     H1-minus-H2 level in the +16 corpus against real female singing at matched f0.
     No formant tracker required. `[prediction]` H1 is systematically low in the
     augmented corpus, and that is a concrete candidate for the thin/buzzy percept.

  `[prediction]` The +16 st corpus (Brandy's median 291 Hz → ~733 Hz, p99 407 Hz →
  ~1026 Hz) will show R1 frozen at its 291 Hz-context value while f0 sits far above
  it — a configuration real voices retune out of, in exactly the band where the model
  fails. No GPU, no training run. A null result kills cluster A outright; a positive
  result makes A1 the priority.
- `F7` **If A1 is attempted, use the real warp, not a uniform scale.** `[idea]` Move R1
  toward f0 (or 2f0 if the target style is belt), leave R2 roughly alone, do not touch
  R3 and above, and saturate near 1 kHz. `[why]` §5.4: R2 rises "proportionally
  smaller" than R1 and only incidentally, R3 "slightly", and rounded vowels cannot
  reach 1 kHz at all. `[gate]` Leaving R3+ untouched is also what preserves the
  identity margin — a uniform warp does not.
- `F8` **Add voiced/unvoiced error to the augmentation acceptance gate.** `[idea]` S3
  found pitch augmentation improved everything **except** VUV_ERR, and attributed it to
  semitone-shift artifacts. This project already has a measured spurious-voicing
  defect. `[cost]` Near zero — a VUV rate is cheaper than every metric already in the
  gate — and there is published precedent that it is the one metric that catches PA
  damage specifically.

---

## 7. What the literature does NOT support

Stated plainly, including the parts that argue against ideas already in flight.

1. **Large-shift offline pitch augmentation as the training-data strategy.** Every
   located paper that tuned the shift magnitude converged on *small*: S3 found ±1 st
   beat ±2 st on all three objective metrics and wrote that too much variation "may
   reduce the effectiveness of the acoustic model"; S2 hard-caps at 5 st because
   "quality … will drop sharply"; S1's own baseline augmentation is ±5 st; S9's
   abstract says artifacts get worse with larger jumps. **Nothing located supports
   building +12/+16 st offline training files.** The one paper that used offline shifts
   at all for range purposes (S5) used ±1 st and dropped it.

2. **Pitch augmentation on its own.** S3 Table 5: adding PA on top of the
   cycle-consistent baseline made MCD and semitone accuracy **worse** (8.06 → 8.11,
   73.67 → 73.46) while VUV_ERR moved the other way; it only paid off combined with
   mix-up. S5: "no significant benefit". Neither is a knock-down result — S3's deltas
   are small, carry no error bars and come from one dataset, and S5 has no table. The
   honest statement is the negative one: **there is no located result where
   pitch-shift augmentation alone clearly helped.**

3. **"Formant-corrected" = scaling the spectral envelope.** The phonetics contradicts
   the transform, not the goal (§5.4). A uniform warp misplaces R2, R3 and the
   singer's-formant region, which is where identity lives and where this project's
   own gate sits.

4. **C1 as written.** No located system uses its own output as a training target.
   S1 goes out of its way not to.

5. **Genuine range extension from zero coverage.** Nothing located addresses a target
   corpus with *no* frames in the required band. S4 is the nearest and it is about
   *sparse* coverage, with its range evidence below D5 (587 Hz) — under this project's
   failure band. Everything else found moves the **source** down rather than extending
   the **target** up. **This is a bounded-search absence, reported as such. It is not
   a novelty claim.**

6. **Nothing at all was located on:** A2 (aperiodicity transplant across singers), A3
   (harmonic-amplitude extrapolation), B2 (register curriculum), C2 (external SVS
   synthesis as training data), C3 (round trip through a third voice), D2
   (register-split rendering). A3, B2, D2 and E2 were not searched for directly and
   are marked not-checked rather than absent.

7. **The strongest positive result here is not about data at all.** S1's gain comes
   from an in-graph training strategy with no augmented files; S4's largest ablation
   is pre-training, not augmentation; S3's best single augmentation is mix-up, not
   pitch shift. The one round-1 idea the literature backs most cleanly (E3) was filed
   under "low prior".

8. **Transfer is assumed throughout.** Every system cited is DDSP+diffusion or
   FastSpeech-style SVS on Mandarin or Japanese corpora, with NSF-HiFiGAN or HiFi-GAN
   vocoders. None is so-vits-svc-fork. None of these numbers were measured on a VITS
   model with a posterior encoder, and none on English pop/R&B. `[assumption]`

9. **The missing perspective flagged in §2 is still missing, and now visibly matters.**
   §5.4 is phonetics, read from two full texts and two abstracts by a non-expert. The
   conclusion that resonance tuning is the unmeasured realism axis is the single most
   load-bearing inference in this document and it has not been checked by anyone who
   works on voice.

---

# 8. Execution log: corpus v5 (2026-09-17)

What was actually built, and every bug the gates caught. Recorded because the
next person will reach for the same shortcuts.

## Bugs in the corpus builder, all caught before the GPU

1. **Instrumental tracks in the training set.** `data/samples/<profile>/**/*.wav`
   also matches `instrumental.wav` - **71 of 149 files are backing tracks with
   no voice in them**. The old builder globbed `sample_*/vocals.wav`; the new one
   did not. Training on this would have poisoned the model with band music, and
   nothing in the training log would have said so.
2. **Every file collides on basename.** All 149 are named `vocals.wav` or
   `instrumental.wav` in per-sample subdirectories. A basename-derived tag made
   1965 writes land as 505 files, each overwriting the last. Tag from the PARENT
   directory (`sample_001`).
3. **The measurements were contaminated too.** The brightness target was first
   computed over the mixed set. On vocals only she is already ON target at
   3-8 kHz; the real hole is ~12 dB across 8-16 kHz. The original design would
   have applied +40 dB of "correction" to 16-20 kHz, where her stems measure
   **-69 dB** - brick-walled, almost certainly lossy-codec or separation loss
   upstream. Nothing can put back a band that was never captured; that band is
   now excluded from both builder and gate.

## Bugs in the training launch

4. `max_epochs` set below the seed's own epoch -> instant exit.
5. Paired G_389 with D_400 because the preserve loop only ever copied `G_*.pth`,
   so a matching `D_389` was assumed not to exist. It did. md5 caught it.
6. CUDA OOM from restarting before the killed process released VRAM. The model
   takes ~28 GB of a 31 GB card, so two overlapping runs cannot fit. Drain to
   <500 MB before relaunching.

## The corpus that passed

| | world33 (what G_389 trained on) | v5 |
|---|---|---|
| files | 398 | 1464 (488 plain + 976 shifted) |
| shifts | +12, +16 st | +4, +8, +12, +16, +20 st, weighted high |
| frames >=700 Hz | ~0 % | **40.7 %** |
| 8-12 kHz vs source | -28.4 dB | -20.7 dB |
| 12-16 kHz vs source | -31.8 dB | -20.2 dB (on target) |
| identity across shifts | - | 0.9857 - 0.9958 |

**The headline is the third row.** The previous run weighted the mel loss toward
high-f0 frames in a corpus whose highest content was 733 Hz, while measuring
whether the model had learned 700-1100 Hz. The augmentation stopped exactly
where the model starts failing.

## Method notes that cost time

- **Verify the shifter on a synthetic tone, never on real audio.** pyin
  octave-halves ~40 % of frames here; a +12 st shift of a real sample read
  189 Hz and looked exactly like the inverted-resample bug returning. The tone
  gate (300 -> 401/601/758/954 Hz) settled it in seconds.
- **A median LTAS lies about bandwidth.** "WORLD truncates fmax to 16.1 kHz" was
  a median reading a sparser harmonic series. By ENERGY the shift costs ~2 dB,
  most of it in the round-trip. Measure bandwidth with energy, not medians.
- **`pre-config` regenerates config.json**, silently restoring `keep_ckpts`.
  Re-apply the config AFTER it. `baseline`/`aug1216` carry `keep_ckpts: 60`, a
  66 GB ceiling.
- **Sample GPU utilisation more than once.** `pre-hubert` alternates GPU and CPU
  phases; a single reading caught a trough and read 0 %, five readings showed
  46/0/0/45/10 %.
