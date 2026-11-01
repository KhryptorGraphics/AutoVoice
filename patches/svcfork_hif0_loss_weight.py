"""Patch so-vits-svc-fork's mel loss to weight high-f0 frames.

Why: measured on the G_250 Hero render, the decoder degrades as Brandy climbs.
Frames more than 6 dB under the source's level, by the SOURCE's f0:

    300-400 Hz   1.9 %        inharmonic excess vs source   +7 %
    400-550 Hz   8.4 %                                     +18 %
    550-700 Hz   8.4 %                                     +11 %
    700-1000 Hz 16.6 %                                     +41 %

The training corpus is NOT the problem - the WORLD +12/+16 st files measure
0.343/0.365 inharmonic at 700-1100 Hz against a real voice's 0.329, i.e. they
are clean. The model is failing to learn data it already has, because
high-f0 frames are a small minority of total frames and the unweighted mel
loss is dominated by the 300-400 Hz bulk (270 plain files vs 128 shifted).

So weight the per-frame mel loss up with f0. The weight is a normalised
WEIGHTED MEAN, not a sum, so the loss keeps the same scale as before and
c_mel / learning_rate keep their meaning - otherwise this would silently act
as a learning-rate change too.

Env-gated so an unset variable is exactly upstream behaviour:
    SVCFORK_HIF0_WEIGHT   max weight at/above HI (default 1 = off)
    SVCFORK_HIF0_LO/HI    ramp start/end in Hz (default 400 / 800)

F0 is interpolated before training; the paired UV mask is authoritative.
Unvoiced frames (uv=0) must retain baseline weight 1.
"""
import re, sys, pathlib

TARGET = "            loss_mel = F.l1_loss(y_mel, y_hat_mel) * self.hparams.train.c_mel\n"
REPL = '''            # SVCFORK_HIF0_WEIGHT: weight mel loss up where the source sings high.
            # Mask interpolated f0 by uv so unvoiced frames retain weight 1.
            _hif0_w = float(os.environ.get("SVCFORK_HIF0_WEIGHT", "1") or 1)
            if _hif0_w > 1.0:
                _hif0_lo = float(os.environ.get("SVCFORK_HIF0_LO", "400"))
                _hif0_hi = float(os.environ.get("SVCFORK_HIF0_HI", "800"))
                _f0_seg = commons.slice_segments(
                    f0, ids_slice, y_hat_mel.shape[-1]
                ).to(y_hat_mel.dtype)
                _uv_seg = commons.slice_segments(
                    uv, ids_slice, y_hat_mel.shape[-1]
                )
                _f0_seg = torch.where(
                    _uv_seg > 0, _f0_seg, torch.zeros_like(_f0_seg)
                )
                _ramp = ((_f0_seg - _hif0_lo) / max(_hif0_hi - _hif0_lo, 1e-6)).clamp(0.0, 1.0)
                _w = (1.0 + (_hif0_w - 1.0) * _ramp).unsqueeze(1)
                _l1 = F.l1_loss(y_mel, y_hat_mel, reduction="none")
                # weighted MEAN: identical scale to the unweighted loss when
                # every weight is 1, so c_mel and the LR keep their meaning.
                loss_mel = ((_l1 * _w).sum() / _w.expand_as(_l1).sum()) * self.hparams.train.c_mel
            else:
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * self.hparams.train.c_mel
'''

def main(path):
    p = pathlib.Path(path)
    src = p.read_text()
    if "SVCFORK_HIF0_WEIGHT" in src:
        if "_uv_seg = commons.slice_segments(" in src:
            print("already patched with uv mask"); return 0
        print("existing HIF0 patch lacks uv mask - refusing to skip it", file=sys.stderr)
        return 4
    if TARGET not in src:
        print("ANCHOR NOT FOUND - refusing to patch", file=sys.stderr); return 2
    if not re.search(r"^import os$", src, re.M):
        print("no 'import os' - refusing", file=sys.stderr); return 3
    p.write_text(src.replace(TARGET, REPL, 1))
    print("patched", path)
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
