# What the SERVED path produces today, measured (2026-10-31)

The arms in `docs/v6-brightness-only-finetune.md` and
`docs/v7-spa-cycle-arm.md` both failed to beat the model that is actually
served. This file records what that served model measures, so the next
person starts from a measured baseline instead of a remembered one.

## The served model

`data/fork_models/_hif0_20260917/G_389.pth`, profile
`fb17af66-8415-4ffe-81b3-600efe75b6d7`, registry at
`data/fork_models/fb17af66-8415-4ffe-81b3-600efe75b6d7.json`:
crepe f0, no auto-predict f0, noise_scale 0.4, db_thresh -100, pad 1.0 s,
chunk 30 s, max chunk 40 s, uv contract ON, crepe periodicity threshold 0.3.

## Measured on the Hero source (`experiments/score_multi.py`, 4 renders)

| band | dropout % | inharmonic excess % |
|---|---|---|
| 250-400 Hz | 2.2 [1.9, 2.6] | 2.7 [1.8, 3.1] |
| 400-550 Hz | 1.2 [1.2, 1.4] | 33.7 [32.9, 34.4] |
| 700-1100 Hz | **35.5** [32.8, 37.5] | **44.9** [42.5, 47.9] |

On the dedicated high-register window (168-188 s, 334 usable frames in the
700-1100 Hz band): dropout 11.4 [9.9, 12.6], inharmonic 69.9 [65.4, 74.0].

Identity (mel-stat centroid) 1.000 on the full song, 0.9999 on the window.
Band energy, relative to 300-3000 Hz: 3-8 kHz -15.9, 8-12 kHz -28.3,
12-16 kHz -34.7, 16-20 kHz -46.2. Centroid 748 Hz.

The 550-700 Hz band holds 30 source frames on the full song and 0 on the
high-register window. It is anecdote, never measurement.

## The remaining defect, in one sentence

**Her corpus is dark and her top register is untrained, so the model
renders her high notes as noisy residue rather than voice.** The 700-1100 Hz
band is where dropout and inharmonicity are both high, and it is where the
served model is worst relative to the source.

## Independent instrument (earshot, full song, served output vs source)

| | source | served vocals |
|---|---|---|
| integrated LUFS | -18.4 | -18.4 |
| true peak dBTP | -3.0 | -4.1 |
| noise floor dBFS | -63.4 | -81.0 |
| brightness Hz | 3991 | 2600 |
| centroid Hz | 1346 | 1026 |
| air band (8-16k) dB | -15.2 | -24.9 |
| top band (16-20k) dB | -30.1 | -46.4 |
| tonal peak ratio dB | 35.4 | 46.0 |
| digital zeros | 0.7 % | 1.6 % |
| clipped samples | 0 | 0 |

Loudness and peak are clean — no clipping, matched LUFS. The defect is
SPECTRAL: the served voice is ~10 dB darker above 8 kHz, its centroid is
320 Hz lower, and its tonal-peak ratio is 10.6 dB higher, which is the
signature of narrowband residue sitting where breath should be. Bandwidth is
not the issue: `effective_fmax` reads 16051 Hz on the served output and
22050 Hz on the source.

On the high-register window the same signature: air -27.7 dB against the
source's -22.1, top -47.1 against -36.9, tonal peak ratio 55.2 against
49.5.

## What this rules out, and what it does not

RULED OUT, with measurements:
- **Level.** The old complaint "the voice isn't steady in volume" is fixed
  in the served path: the dropout fill runs last, and the measured dropout
  at 250-400 Hz is 2.2 % and at 400-550 Hz 1.2 %.
- **Clipping and loudness.** 0 clipped samples, true peak -4.1 dBTP, LUFS
  matched to the source.
- **Bandwidth.** `effective_fmax` 16051 Hz, not a narrow wall.

NOT RULED OUT:
- **The high register.** 35.5 % dropout and 44.9 % inharmonic excess at
  700-1100 Hz, on 424 source frames. Every training-data lever tried has
  failed to move it (v5 pitch-shift augmentation, v6 brightness-only, the
  SPA-SVC cycle loss). The architectural question is now open: ContentVec's
  8 kHz Nyquist leaves the decoder with no content input above 8 kHz, and
  the NSF excitation is fitted on a 291 Hz median. Whether this register is
  representable at all is the open question (AV-41e4).
- **The HF residue.** The 10.6 dB tonal-peak-ratio excess is the "double
  ringing around the voice" complaint, and it is the same residue that
  fills a dropout. Raising brightness makes it worse (v6), which is why
  the output-side air restore is capped.

## The next lever that does not depend on the training corpus

AV-9hc5: pitch-tracked make-up gain plus keep-source-spans above ~550 Hz.
It is signal processing, so it is not gated on whether the corpus can
represent her high register.

## Reproducing these numbers

    # renders (4 per model, the serving flags)
    python scratchpad/v6eval/render_candidates.py base389=<path to G_389.pth>

    # score
    python experiments/score_multi.py --source scratchpad/dropout/hero_src.wav \
        --group base389='scratchpad/v6eval/base389_r*.wav' \
        --json scratchpad/v6eval/arm_scores.json

    # band limit, with the project's authority function
    PYTHONPATH=src python -c "
    import soundfile as sf
    from auto_voice.inference.fork_voice_polish import effective_fmax
    y, sr = sf.read('<file>', dtype='float32')
    print(effective_fmax(y[:int(sr*60)], sr))"
