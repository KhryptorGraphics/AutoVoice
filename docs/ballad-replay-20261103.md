# Famous-ballad replay, and what the inharmonic metric was really measuring (2026-11-03)

Brandy (`fb17af66`, G_389) on 13 songs, scored per song against that song's own
separated vocal (the "key": what the decoder converted and every chain stage keys
on). The song set: Whitney Houston *I Will Always Love You* and *I Have Nothing*,
Beyoncé *Listen*, Celine Dion *My Heart Will Go On* and *Because You Loved Me*,
Alicia Keys *If I Ain't Got You*, Christina Aguilera *Beautiful*, Adele *Someone
Like You*, Toni Braxton *Un-Break My Heart*, Kelly Clarkson *Because of You*
(all YouTube), Samantha Harvey *All of Me (1 Mic 1 Take)*, an R&B cover of
*Because You Loved Me*, and Hero.

Each song went through exactly what a served single-stem job does - the service's
separation (htdemucs_ft), the fork render through the bridge, `_match_lead_level`
and `_correct_fork_output` (studio preset, effort 1.0) - with key, raw render and
final kept (`scratchpad/ballads/replay_song.py`, `run_batch.sh`, `score_songs.py`).
A stage-by-stage capture of the same chain reproduces the batch's final bit for
bit (max |diff| 0), so every stage number below is production behaviour.

## 1. The inharmonic metric measured spectral tilt, not noise

`experiments/score_multi.py` counted a frame's energy as harmonic only within ±3 %
of k·f0, k ≤ 11. The Hann main lobe of a 2048-point STFT is ±2 bins (43 Hz), so for
every harmonic below 1.43 kHz the window was narrower than the harmonic's own
energy: the skirt of a perfectly clean partial counted as "inharmonic", and
anything that changed how loud the lowest harmonics are moved the number with no
noise added.

Proof on known content - a pure harmonic tone plus -35 dB white noise, then a
time-invariant boost that adds no noise (old window / true between-harmonic share
with a ±max(3 %, 2 bins) window):

| f0 | as is | +6 dB below 1.5·f0 | +6 dB above 4·f0 |
|---|---|---|---|
| 180 Hz | 0.642 / 0.023 | **0.702** / 0.024 | 0.632 / 0.021 |
| 300 Hz | 0.400 / 0.031 | **0.426** / 0.029 | 0.414 / 0.033 |
| 450 Hz | 0.248 / 0.043 | **0.302** / 0.039 | 0.242 / 0.045 |
| 800 Hz | 0.140 / 0.066 | **0.164** / 0.060 | 0.135 / 0.071 |

At 180 Hz the old metric reported 28x the true share, and a louder fundamental
alone raised it 9 %. **score_multi now scores with ±max(3 %, 2 bins) and every
harmonic up to 6 kHz.** Its `--demo` self-check gained a regression check that
fails on the old window (a +6 dB fundamental moved 0.1620 -> 0.1785) and passes
on the new one (0.0348 -> 0.0363, while added noise reads 0.0996). Inharmonic
numbers scored before 2026-11-03 are not comparable with ones scored after.
`static_metrics.py`'s `offharm` uses ±max(3 %, 1 bin): less affected, unchanged.

**AV-n6o4 is settled by this.** The effort follower's "250-400 Hz cost" (2.4 ->
5.0 % on the old window) is the artifact. Captured stage by stage on two ballads,
the follower is the only stage that moves the old window below 400 Hz, and on the
main-lobe window it *lowers* between-harmonic excess vs the key (whitney_ihn
150-250 Hz +59.5 -> +34.2 %, 250-400 Hz +35.1 -> +8.3 %; celine_mhwgo +129.9 ->
+73.0 % and +62.7 -> +50.2 %). The subharmonic stage was the other suspect (a yin
octave-up error would put its "half-integer partial" cuts on real harmonics): its
yin reads >= 99 % of 150-550 Hz frames at the right octave on four songs, none an
octave up.

## 2. The valid ranking across 13 songs

Between-harmonic share relative to the key, raw render -> production final, %
(`scratchpad/ballads/wide_inharm.py`; "–" = under 25 frames in the band):

| song | 150-250 | 250-400 | 400-550 | 550-700 | 700-1100 |
|---|---|---|---|---|---|
| adele_sly | +117 → +107 | +114 → +110 | +73 → +5 | +51 → +8 | – |
| aguilera_beautiful | +63 → +49 | +79 → +74 | -0 → -12 | -9 → -16 | – |
| beyonce_listen | +103 → +77 | +62 → +43 | +37 → -4 | +56 → +2 | – |
| braxton_ubmh | +153 → +136 | +52 → +56 | +34 → -2 | -36 → -40 | +16 → -4 |
| celine_bylm | +67 → +70 | +27 → +16 | +82 → -1 | -1 → -6 | +13 → -5 |
| celine_mhwgo | +121 → +73 | +56 → +50 | +65 → +9 | – | +107 → +3 |
| clarkson_boy | +107 → +83 | +96 → +85 | +66 → +12 | +77 → +7 | – |
| cover_bylm | +91 → +60 | +70 → +66 | +69 → -7 | +77 → +6 | +43 → +2 |
| harvey_allofme | +90 → +85 | +81 → +80 | +28 → +14 | – | – |
| hero | +105 → +73 | +72 → +77 | +45 → +10 | +34 → +0 | +61 → +8 |
| keys_iagy | +62 → +58 | +25 → +24 | +37 → -7 | -3 → -9 | – |
| whitney_ihn | +50 → +34 | +27 → +8 | +0 → -24 | +43 → +6 | +61 → +10 |
| whitney_iwaly | +60 → +46 | +33 → +20 | +57 → +1 | +27 → -5 | +54 → +11 |

**The high register is at the key.** After the chain no song sits more than 14 %
above its key anywhere from 400 to 1100 Hz (the lowest, braxton at 550-700 Hz, is
40 % *cleaner* than its key, as its raw render already was); the inharmonic notch
does that work (raw renders reach +107 % there).

**The low register's excess is partly Brandy's own voice.** Read against the key,
150-400 Hz looks like 1.5-2.4x the source's between-harmonic energy, but the target
is breathier than these sources. Her profile recordings
(`data/samples/fb17af66-.../sample_*/vocals.wav`, separated stems - the same kind of
signal as a key), gated and windowed identically
(`scratchpad/ballads/target_share_perfile.py`, corrected 2026-11-06), give her own
level. Only samples recorded as hers count: 001-064 are clips of brandy.mp4 and 065
is brandy_audio.wav (12.8 min at 16 kHz; 066 is a byte-identical copy). Left out:
078 is a recording made on the Sing Along page while Hero played and attached to
her profile (the subharmonic measure does not flag it; who sang it, and whether the
backing track bleeds in, is not recorded), 067 is a 16 kHz
"booktalk" file, 068-076 are Video_*/b3 uploads of unrecorded provenance (068 and
070 yield no usable frames; 070/074 and 076/077 are the same performances). Each of
her two sources on its own:

| band | her share, median: brandy.mp4 / brandy_audio.wav | 13 keys, median | 13 finals, median (range) | finals vs her median |
|---|---|---|---|---|
| 150-250 Hz | 0.143 / 0.146 (756 / 765 frames) | 0.088 | 0.151 (0.132-0.215) | +4 to +6 %; 10-11 of 13 above |
| 250-400 Hz | 0.148 / 0.150 (4648 / 4722 frames) | 0.117 | 0.178 (0.155-0.203) | +18 to +20 %; 13 of 13 above |

At 150-250 Hz the conversion sits at her own level. At 250-400 Hz about half of the
excess over the source is her timbre and the rest is the conversion: every final is
breathier than her own recordings, by about a fifth - inside the spread of her own
frames (p25-p75 0.111-0.203, both sources) but systematic. (The raw renders already
carry it: 0.184 median at 250-400 Hz, before the chain.) The first version of this
section pooled all 78 samples (0.138 / 0.143) and
called the whole excess hers; that overstated it.

Above 550 Hz her own recordings give no reference: 8 gated frames at 550-700 Hz and
none higher. The full pool's 142 and 73 frames at 550-700 and 700-1100 Hz are mostly
b3/Video_1, _2 and _5 (129 and 66 of them).

Dropout after the chain is <= 1.2 % in every band of every song except 150-250 Hz
on celine_bylm and cover_bylm (3.1 %: 7 and 15 frames).

## 3. Register folding: rendered in range, restored by comb - rejected

The creative arm for AV-41e4's out-of-range top notes. Render the same key again at
-12 st, so her top notes land where the model was trained (700-1000 Hz -> 350-500
Hz), then restore the octave on the render itself: with T the -12 render's period,
`(B(t) + B(t - T/2)) / 2` cancels every odd harmonic of f0/2 and keeps the even
ones, which sit exactly on the target's harmonics m·f0 with the formant envelope
untouched - no vocoder, no resynthesis (the WORLD octave-up tried 2026-11-01 failed
by resynthesising at the high pitch). Phrases holding >= 3 frames >= 500 Hz were
swapped in the key's unvoiced gaps, then the whole song ran the production chain
(`scratchpad/fold/fold.py`).

- The restore works: on the folded spans the restored render sits at the key's
  pitch on 69.7 % of key-voiced frames, the 0 st render on 71.8 % (the -12 render:
  66.4 % an octave below, as rendered).
- Before the chain it does what it was built for (Hero draw 0, old window):
  700-1100 Hz dropout 50.5 -> 19.3 %, inharmonic +45.9 -> +21.5 %; 400-550 Hz
  +24.4 -> +29.9 %, because whole phrases were folded, so their 400-550 Hz notes
  were rendered an octave down too.
- After the chain, paired on Hero's four production draws (same key per draw,
  main-lobe window): 700-1100 Hz between-harmonic excess +7.6..+13.1 -> +8.6..+13.4 %
  (no better), 400-550 Hz +5.7..+9.8 -> +15.1..+16.5 %, 550-700 Hz -11.9..-3.2 ->
  +6.9..+11.1 % (worse on all four). The only gain: 700-1100 Hz dropout 0-0.42 % ->
  0 % (1-2 frames per render). The chain already does to the 0 st render what the
  fold does before it, and the fold doubles the render cost.

## 4. The notch cuts the harmonics' own skirts

`fork_inharmonic_notch` copies the old metric's ±3 % window, so the same flaw acts
on audio. Splitting the energy it removes on its own frames (production chain,
`scratchpad/ballads/notch_skirt.py`):

| song | harmonic skirts (±3 % .. ±2 bins) | inside ±3 % | between harmonics |
|---|---|---|---|
| hero | 26.2 % | 34.4 % | 39.4 % |
| whitney_iwaly | 37.4 % | 49.3 % | 13.3 % |
| celine_mhwgo | 38.7 % | 27.3 % | 34.0 % |
| cover_bylm | 36.8 % | 35.9 % | 27.3 % |

Cutting a partial's skirt bins and resynthesising lowers its peak too, so 61-87 %
of what the stage removes is harmonic energy; the make-up gain then lifts the whole
frame back. On a clean vibrato tone whose fundamental is 6 dB stronger than the
key's - no junk at all - the stage cuts 0.73 dB and imposes 4.4 % amplitude
modulation at vibrato rate on the fundamental (a sub-bin mask gating a sweeping
partial in and out).

A main-lobe window (±max(3 %, 2 bins)) fixes that on synthetic signals outright:

| case | ±3 % (production) | main-lobe window |
|---|---|---|
| clean tone, fundamental +6 dB vs key, 600 Hz | -0.73 dB, AM 4.41 % | -0.00 dB, AM 0.00 % |
| half-integer junk at -12 dB, 600 Hz | junk -11.9 -> -18.1 dB, AM 3.45 % | -11.9 -> -20.7 dB, AM 0.24 % |
| junk + fundamental +6 dB, 800 Hz | -16.8 -> -25.0 dB, AM 2.85 % | -16.8 -> -26.5 dB, AM 0.26 % |

**On real renders it is a trade-off, so production keeps ±3 %.** Paired on all 13
songs' cached raw renders (`scratchpad/ballads/notch_wide_ab.py`), production ->
main-lobe window, between-harmonic excess vs the key, %:

| song | 400-550 | 550-700 | 700-1100 | fast harmonic AM, dB rms: key / raw / prod / lobe |
|---|---|---|---|---|
| adele_sly | +5 → +11 | +8 → +14 | – | 1.68 / 1.57 / 1.65 / 1.56 |
| aguilera_beautiful | -12 → -6 | -16 → -10 | – | 2.14 / 0.99 / 1.11 / 1.04 |
| beyonce_listen | -4 → +3 | +2 → +14 | – | 2.35 / 1.38 / 1.53 / 1.46 |
| braxton_ubmh | -2 → -10 | -40 → -38 | -4 → -6 | 1.98 / 1.72 / 1.80 / 1.76 |
| celine_bylm | -1 → +9 | -6 → -8 | -5 → -5 | 1.81 / 1.75 / 1.52 / 1.48 |
| celine_mhwgo | +9 → +14 | – | +3 → -1 | 1.92 / 1.83 / 1.70 / 1.69 |
| clarkson_boy | +12 → +2 | +7 → +0 | – | 1.88 / 1.60 / 1.62 / 1.61 |
| cover_bylm | -7 → +0 | +6 → +16 | +2 → +20 | 1.71 / 1.57 / 1.53 / 1.49 |
| harvey_allofme | +14 → -9 | – | – | 1.82 / 1.23 / 1.24 / 1.23 |
| hero | +10 → +7 | +0 → +12 | +8 → +2 | 1.93 / 2.08 / 1.65 / 1.67 |
| keys_iagy | -7 → +11 | -9 → -3 | – | 1.50 / 1.22 / 1.40 / 1.35 |
| whitney_ihn | -24 → -13 | +6 → +15 | +10 → +13 | 2.04 / 1.57 / 1.67 / 1.59 |
| whitney_iwaly | +1 → -4 | -5 → -3 | +11 → -3 | 1.64 / 1.12 / 1.20 / 1.15 |

Fast AM (harmonics 1-3 on gated runs >= 400 Hz, dB minus its 105 ms moving
average) falls on 12 of 13 songs, but only by 0.04 dB rms (median), against the
singers' own 1.5-2.4. Between-harmonic energy rises at 550-700 Hz on 9 of 11
(median +6 points) and at 400-550 Hz on 8 of 13. The reason is in the stage's maths:
it solves for the cut that brings the render's off-harmonic fraction to the key's,
and counting skirts as "off" inflated that fraction, so the production cut is
deeper and removes more real junk along with the skirts - it already lands 400-550
Hz below the key on 7 songs (whitney_ihn -24 %). The main-lobe window corrects
toward the key's true share and stops there. Dropout moves under 0.5 points in
every band; belt off-harmonic share +0.004 (up on 9); the HF shares move -0.12 dB
or less on average, level slope +0.02.
Which reads better - production's deeper, skirt-cutting notch or the gentler
main-lobe one - is a listening question; each song's `final_lobe.wav` sits next
to its `final.wav` in `scratchpad/ballads/replay/<song>/`.

## For listening

The ten YouTube ballads, and a second draw of the *Because You Loved Me* cover,
went through the live service one at a time on 2026-11-03 (studio preset,
Brandy, 13:00-14:10, about 6 min each); all eleven completed and are in History.
Every mix lands at -1.5 to -1.8 dBTP with 0 clipped samples, every vocal stem at
-2.1 to -6.0 dBTP with 0 clipped: the replay finals' +3-4.6 dBFS peaks above are
before the served path's peak guards.

| song | job |
|---|---|
| Whitney Houston - I Will Always Love You | `a2561c15` |
| Beyoncé - Listen | `0bfd3307` |
| Celine Dion - My Heart Will Go On | `fa9f173a` |
| Whitney Houston - I Have Nothing | `dc1103d8` |
| Alicia Keys - If I Ain't Got You | `240547d4` |
| Christina Aguilera - Beautiful | `13b034ad` |
| Adele - Someone Like You | `b1e2bf3b` |
| Toni Braxton - Un-Break My Heart | `3fd6bff9` |
| Kelly Clarkson - Because of You | `f687d287` |
| Celine Dion - Because You Loved Me | `933d3c73` |
| *Because You Loved Me* R&B cover (second draw) | `38b4614e` |

Converted earlier: All of Me, Samantha Harvey 1 Mic 1 Take (`706cf70f`), and
the first draw of the cover (`65dbffa1`).
