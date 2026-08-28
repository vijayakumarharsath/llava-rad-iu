#!/usr/bin/env python3
# patch_train_sampler.py -- one-shot, env-gated WeightedRandomSampler patch
# for ~/LLaVA-Rad/llava/train/train.py.
#
# WHY: v2 training oversamples abnormal reports (weights in
# ~/iu-xray/v2_weights.json from prep_v2.py). This fork builds its
# dataloader INSIDE the HF Trainer (there is no explicit DataLoader in
# train.py), so the patch inserts a small Trainer subclass immediately
# before `trainer = LLaVATrainer(...)` in train(). The whole block is
# gated on the V2_WEIGHTS env var: with the env var unset the file
# behaves exactly like before (v1 reruns are unaffected).
#
# What the inserted block does:
#   - if $V2_WEIGHTS is set: reads the weights list, asserts its length
#     equals len(train_dataset) (2837), defines
#     _WeightedLLaVATrainer(LLaVATrainer) overriding
#     get_train_dataloader() to use
#     WeightedRandomSampler(weights, num_samples=len(weights),
#                           replacement=True)
#     -> the sampler draws exactly len(dataset) samples per epoch, so
#        the iteration count is UNCHANGED (~177 iters/epoch, ~3h20m);
#        abnormal reports simply get drawn ~2x more often per epoch
#   - the existing `trainer = LLaVATrainer(...)` line is rewritten to
#     instantiate `_WeightedLLaVATrainer`, which the if/else above binds
#     to the subclass (env set) or to the original class (env unset,
#     i.e. v1 reruns are bit-for-bit the same code path)
#   NOTE: we deliberately do NOT rebind the name LLaVATrainer inside
#   train() -- an assignment to a name that is also read in the same
#   function makes it LOCAL function-wide, which would break the class
#   base reference (UnboundLocalError). The fresh name avoids that.
#
# Single-GPU recipe only (v2 trains with CUDA_VISIBLE_DEVICES=1, no
# deepspeed/DDP), which is the simple get_train_dataloader path in
# transformers 4.40.2.
#
# Safe by construction:
#   - REFUSES to run if the file already contains "V2_WEIGHTS" (idempotent)
#   - REFUSES if the exact anchor line is not found (file changed?)
#   - ast.parse()s the patched source before writing anything
#   - writes a .bak-sampler backup of the original first
#   - prints the patched region for eyeballing
#
# Usage:
#   python3 patch_train_sampler.py            # patches the machine default
#   python3 patch_train_sampler.py <path>     # or an explicit train.py path
#
# After patching, training starts with:
#   V2_WEIGHTS=~/iu-xray/v2_weights.json ... (set inside run_train_v2.sh)

import ast
import os
import sys

ANCHOR = "    trainer = LLaVATrainer(model=model,"
ANCHOR_V2 = "    trainer = _WeightedLLaVATrainer(model=model,"

BLOCK = '''    # ---- v2 weighted sampling (abnormal oversampling), env-gated ----
    if os.environ.get("V2_WEIGHTS"):
        import json as _json
        from torch.utils.data import DataLoader, WeightedRandomSampler
        _w2 = _json.load(open(os.environ["V2_WEIGHTS"]))["weights"]
        assert len(_w2) == len(data_module["train_dataset"]), \\
            "v2_weights length mismatch: %d vs %d" % (
                len(_w2), len(data_module["train_dataset"]))
        class _WeightedLLaVATrainer(LLaVATrainer):
            def get_train_dataloader(self):
                ds = self._remove_unused_columns(self.train_dataset, None)
                return DataLoader(
                    ds,
                    batch_size=self._train_batch_size,
                    sampler=WeightedRandomSampler(_w2, num_samples=len(_w2),
                                                  replacement=True),
                    collate_fn=self.data_collator,
                    num_workers=self.args.dataloader_num_workers,
                    pin_memory=self.args.dataloader_pin_memory)
        print("v2 weighted sampler active: %d weights, %d abnormal" % (
            len(_w2), sum(1 for w in _w2 if w > 1.0)))
    else:
        _WeightedLLaVATrainer = LLaVATrainer
    # ---- end v2 weighted sampling ----
'''


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
        "~/LLaVA-Rad/llava/train/train.py")
    src = open(path).read()
    if "V2_WEIGHTS" in src:
        print("REFUSED: file already contains V2_WEIGHTS "
              "(already patched, or manual edits). Nothing written.")
        sys.exit(1)
    if ANCHOR not in src:
        print("REFUSED: anchor line not found (did train.py change?). "
              "Anchor searched:")
        print(repr(ANCHOR))
        sys.exit(1)
    out = src.replace(ANCHOR, BLOCK + ANCHOR_V2, 1)
    ast.parse(out)  # must still be valid python before we touch the file
    backup = path + ".bak-sampler"
    if not os.path.exists(backup):
        with open(backup, "w") as f:
            f.write(src)
        print("backup -> " + backup)
    with open(path, "w") as f:
        f.write(out)
    print("patched " + path)
    lines = out.splitlines()
    i = [n for n, l in enumerate(lines)
         if "v2 weighted sampler active" in l][0]
    print("--- patched region (eyeball it) ---")
    for l in lines[max(0, i - 18):i + 5]:
        print(l)


if __name__ == "__main__":
    main()
