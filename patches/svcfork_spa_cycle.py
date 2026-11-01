"""Install an opt-in SPA-SVC-like cycle loss in so-vits-svc-fork/train.py.

Patch the installed fork only while no training process is using its source.
The cycle objective is disabled unless SVCFORK_SPA_CYCLE_WEIGHT is positive.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

PATCH_MARKER = "SVCFORK_SPA_CYCLE_WEIGHT"
RUNTIME_MARKER = 'AUTOVOICE_SPA_CYCLE_RUNTIME = "v1"'
IMPORT_ANCHOR = "from .modules.synthesizers import SynthesizerTrn\n"
INIT_ANCHOR = "        self.automatic_optimization = False\n"
LOSS_ANCHOR = (
    "            loss_lf0 = F.mse_loss(pred_lf0, lf0)\n"
    "            loss_gen_all = loss_gen + loss_fm + loss_mel + "
    "loss_kl + loss_lf0\n"
)
# The upstream assignment the replacement must re-emit verbatim. Dropping it
# leaves the patched file referring to an undefined name at the first training
# step - a NameError that neither a syntax check nor a local-variable test
# can see.
LF0_ASSIGNMENT = "            loss_lf0 = F.mse_loss(pred_lf0, lf0)\n"
# A token that only the CURRENT init block contains. Its absence means the
# installed train.py was patched by an EARLIER version of this patcher, so
# the fork would honour an older configuration than this repo describes.
INIT_MARKER = "SVCFORK_SPA_ENCODER_DEVICE"

INIT_BLOCK = "\n".join(
    (
        '        # Opt-in SPA cycle owns a frozen ContentVec model.',
        '        self._spa_cycle_weight = float(',
        '            os.environ.get("SVCFORK_SPA_CYCLE_WEIGHT", "0") or 0',
        '        )',
        '        if self._spa_cycle_weight < 0:',
        '            raise ValueError(',
        '                "SVCFORK_SPA_CYCLE_WEIGHT must be nonnegative"',
        '            )',
        '        self.spa_cycle_objective = None',
        '        if self._spa_cycle_weight > 0:',
        '            _spa_final_proj = bool(',
        '                self.hparams.data.get("contentvec_final_proj", True)',
        '            )',
        '            _spa_f0_ceiling = float(',
        '                os.environ.get("SVCFORK_SPA_F0_CEILING_HZ", "1100")',
        '            )',
        '            _spa_max_samples = int(',
        '                os.environ.get("SVCFORK_SPA_MAX_SAMPLES", "4") or 4',
        '            )',
        '            _spa_encoder_device = os.environ.get(',
        '                "SVCFORK_SPA_ENCODER_DEVICE", "cpu"',
        '            )',
        '            self.spa_cycle_objective = SPACycleObjective(',
        '                content_model=utils.get_hubert_model(',
        '                    _spa_encoder_device, _spa_final_proj',
        '                ),',
        '                sample_rate=int(self.hparams.data.sampling_rate),',
        '                legacy_final_proj=_spa_final_proj,',
        '                f0_ceiling_hz=_spa_f0_ceiling,',
        '                max_cycle_samples=_spa_max_samples,',
        '                encoder_device=_spa_encoder_device,',
        '            )',
        '            LOG.info(',
        '                "SPA cycle enabled: weight=%s ceiling_hz=%s "',
        '                "final_proj=%s max_samples=%s encoder=%s",',
        '                self._spa_cycle_weight,',
        '                _spa_f0_ceiling,',
        '                _spa_final_proj,',
        '                _spa_max_samples,',
        '                _spa_encoder_device,',
        '            )',
    )
) + "\n"

LOSS_REPLACEMENT = "\n".join(
    (
        # The anchor this replaces ALSO contained the loss_lf0 assignment.
        # Dropping it left the patched file referring to an undefined name -
        # a NameError at the first training step, which a syntax check and a
        # local-variable test both miss.
        '            loss_lf0 = F.mse_loss(pred_lf0, lf0)',
        '            if self.spa_cycle_objective is None:',
        '                # Disabled path retains upstream loss arithmetic.',
        '                loss_gen_all = (',
        '                    loss_gen + loss_fm + loss_mel + loss_kl + loss_lf0',
        '                )',
        '            else:',
        '                _spa_frames = y_hat_mel.shape[-1]',
        '                _spa_content = commons.slice_segments(',
        '                    c, ids_slice, _spa_frames',
        '                )',
        '                _spa_f0 = commons.slice_segments(',
        '                    f0, ids_slice, _spa_frames',
        '                )',
        '                _spa_uv = commons.slice_segments(',
        '                    uv, ids_slice, _spa_frames',
        '                )',
        '                _spa_cycle_raw = self.spa_cycle_objective(',
        '                    net_g=self.net_g,',
        '                    content=_spa_content,',
        '                    f0=_spa_f0,',
        '                    uv=_spa_uv,',
        '                    g=g,',
        '                    target_mel=y_mel,',
        '                    mel_spectrogram_fn=mel_spectrogram_torch,',
        '                    hparams=self.hparams,',
        '                )',
        '                loss_spa_cycle = (',
        '                    _spa_cycle_raw',
        '                    * self._spa_cycle_weight',
        '                    * self.hparams.train.c_mel',
        '                )',
        '                loss_gen_all = (',
        '                    loss_gen + loss_fm + loss_mel + loss_kl + loss_lf0',
        '                    + loss_spa_cycle',
        '                )',
        '                # Raw MSE, unweighted: the weighted number cannot be read',
        '                # against the mel loss without knowing the weight.',
        '                # Detached: these are log values, not part of the graph.',
        '                self.log_(',
        '                    "loss/g/spa_cycle_raw", _spa_cycle_raw.detach()',
        '                )',
        '                self.log_(',
        '                    "loss/g/spa_cycle", loss_spa_cycle.detach()',
        '                )',
    )
) + "\n"


def _undefined_names(source: str, names) -> set:
    """Names the source never binds anywhere.

    A syntax check cannot see a NameError, and the fork's loss block is
    assembled by string replacement - so a replacement that drops an
    assignment leaves a name undefined and the file dies at the first
    training step, long after the patch was applied. This catches that class
    of patch bug before the file is written.
    """
    import ast

    bound = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.comprehension,)):
            for inner in ast.walk(node.target):
                if isinstance(inner, ast.Name):
                    bound.add(inner.id)
    return {name for name in names if name not in bound}


def _atomic_write(path: Path, content: str) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def main(path: str) -> int:
    train_path = Path(path)
    if not train_path.is_file():
        print(f"train.py not found: {train_path}", file=sys.stderr)
        return 1
    runtime_source = Path(__file__).with_name("svcfork_spa_cycle_runtime.py")
    if not runtime_source.is_file():
        print(f"runtime helper not found: {runtime_source}", file=sys.stderr)
        return 2

    source = train_path.read_text(encoding="utf-8")
    runtime_text = runtime_source.read_text(encoding="utf-8")
    if RUNTIME_MARKER not in runtime_text:
        print("runtime helper marker missing - refusing to patch", file=sys.stderr)
        return 3

    module_path = train_path.with_name("autovoice_spa_cycle.py")
    if PATCH_MARKER in source:
        # train.py is already patched. Two things can still be wrong with it,
        # and both must be repaired rather than reported as "already patched":
        #
        # 1. The HELPER is a separate file and can drift from the repo source
        #    - a changed helper means the installed fork would train
        #    different semantics than this repo describes, with no error
        #    anywhere.
        # 2. train.py itself can carry the marker and still be BROKEN: an
        #    earlier replacement dropped the `loss_lf0 = ...` assignment, so
        #    the patched file raised NameError at the first training step. The
        #    marker alone does not mean the patch is intact.
        #
        # So: verify the patched file's names and its loss region, and rebuild
        # train.py from the backup when either check fails.
        installed = (
            module_path.read_text(encoding="utf-8")
            if module_path.is_file()
            else None
        )
        helper_ok = installed == runtime_text
        missing = _undefined_names(source, (
            "loss_gen", "loss_fm", "loss_mel", "loss_kl", "loss_lf0",
            "pred_lf0", "lf0", "y_hat_mel", "y_mel", "c", "f0", "uv", "g",
            "ids_slice", "commons", "mel_spectrogram_torch", "F",
        ))
        train_ok = not missing and LF0_ASSIGNMENT in source
        # A name check cannot see a STALE init block: the marker and every
        # name are present, but the block predates a later change (the
        # encoder-device wiring) and the installed fork would then honour an
        # older configuration than this repo describes. So the block's own
        # content is checked too.
        if train_ok and INIT_MARKER not in source:
            train_ok = False
            missing = set(missing) | {"the current init block"}
        if helper_ok and train_ok:
            print("already patched with SPA cycle helper")
            return 0
        if not train_ok:
            detail = ", ".join(sorted(missing)) or "the loss_lf0 assignment"
            print(
                "installed train.py carries the SPA marker but is BROKEN "
                f"(missing: {detail}) - restoring the pre-SPA backup and "
                "re-patching",
                file=sys.stderr,
            )
            backup = _find_backup(train_path)
            if backup is None:
                print(
                    "no pre-SPA backup found - refusing to guess; restore "
                    "train.py by hand and re-run",
                    file=sys.stderr,
                )
                return 9
            shutil.copyfile(backup, train_path)
            source = train_path.read_text(encoding="utf-8")
            if PATCH_MARKER in source:
                print(
                    "the backup is itself patched - refusing to rebuild",
                    file=sys.stderr,
                )
                return 9
            # Fall through to the fresh-patch path below with the restored
            # source. The helper is rewritten by that path.
            return _apply_to_source(
                train_path, source, module_path, runtime_text)
        if not helper_ok:
            # train.py is healthy; only the helper drifted. Rewrite the helper
            # and leave train.py alone - re-patching a patched file would
            # find no anchors to replace.
            print(
                "installed SPA helper differs from the repo source - "
                "refreshing it",
                file=sys.stderr,
            )
            _atomic_write(module_path, runtime_text)
            os.chmod(module_path, 0o644)
            print(f"refreshed {module_path}")
            return 0
    if "autovoice_spa_cycle" in source or module_path.exists():
        print(
            "partial or conflicting SPA cycle installation - refusing",
            file=sys.stderr,
        )
        return 4

    for anchor in (IMPORT_ANCHOR, INIT_ANCHOR, LOSS_ANCHOR):
        if source.count(anchor) != 1:
            print(
                "expected exactly one training anchor, found "
                f"{source.count(anchor)}: {anchor!r}",
                file=sys.stderr,
            )
            return 5
    if "from .autovoice_spa_cycle import SPACycleObjective" in source:
        print("cycle import exists without patch marker - refusing", file=sys.stderr)
        return 4
    return _apply_to_source(train_path, source, module_path, runtime_text)


def _find_backup(train_path: Path):
    """The pre-SPA copy of train.py, and only that one.

    The package directory holds several `train.py.pre_*` files from earlier
    patches, including `pre_hif0_brightness_20261031` and `pre_keyshift`.
    Restoring one of THOSE would silently revert the high-f0 mel weighting's
    uv mask - which the v6 arm depends on - so only the SPA-era backup counts.
    """
    for candidate in sorted(train_path.parent.glob("train.py.pre_spa_cycle_*")):
        return candidate
    return None


def _apply_to_source(train_path: Path, source: str, module_path: Path,
                     runtime_text: str) -> int:
    """Patch, verify, and write. Every check happens BEFORE the write."""
    for anchor in (IMPORT_ANCHOR, INIT_ANCHOR, LOSS_ANCHOR):
        if source.count(anchor) != 1:
            print(
                "expected exactly one training anchor, found "
                f"{source.count(anchor)}: {anchor!r}",
                file=sys.stderr,
            )
            return 5
    patched = source.replace(
        IMPORT_ANCHOR,
        IMPORT_ANCHOR + "from .autovoice_spa_cycle import SPACycleObjective\n",
        1,
    )
    patched = patched.replace(INIT_ANCHOR, INIT_BLOCK + INIT_ANCHOR, 1)
    patched = patched.replace(LOSS_ANCHOR, LOSS_REPLACEMENT, 1)
    if PATCH_MARKER not in patched or "loss_spa_cycle" not in patched:
        print("patch verification failed before write", file=sys.stderr)
        return 6
    try:
        compile(patched, str(train_path), "exec")
    except SyntaxError as exc:
        print(f"patched source does not compile: {exc}", file=sys.stderr)
        return 7
    # A syntax check cannot see a NameError: the patched block can reference a
    # name the replacement dropped (it did once - loss_lf0), and the file only
    # fails at the first training step. So check that every name the injected
    # block USES is still DEFINED somewhere in the patched source.
    missing = _undefined_names(patched, (
        "loss_gen", "loss_fm", "loss_mel", "loss_kl", "loss_lf0",
        "pred_lf0", "lf0", "y_hat_mel", "y_mel", "c", "f0", "uv", "g",
        "ids_slice", "commons", "mel_spectrogram_torch", "F",
    ))
    if missing:
        print(
            "patched source references names it no longer defines: "
            + ", ".join(sorted(missing)),
            file=sys.stderr,
        )
        return 8
    if LF0_ASSIGNMENT not in patched:
        print(
            "patched source no longer contains the upstream loss_lf0 "
            "assignment - refusing to write",
            file=sys.stderr,
        )
        return 8

    _atomic_write(module_path, runtime_text)
    os.chmod(module_path, 0o644)
    try:
        _atomic_write(train_path, patched)
        # mkstemp creates 0600; the fork's files are 0644, and a
        # world-unreadable train.py breaks any other user running the env.
        os.chmod(train_path, 0o644)
    except BaseException:
        module_path.unlink(missing_ok=True)
        raise
    print(f"patched {train_path} and installed {module_path}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            f"usage: {sys.argv[0]} /path/to/so_vits_svc_fork/train.py",
            file=sys.stderr,
        )
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
