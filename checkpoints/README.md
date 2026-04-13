# Checkpoints

This folder stores released checkpoint artifacts for result verification.

## Release protocol
1. Add checkpoint files (e.g., `pems08_best.pth`, `pems_bay_best.pth`).
2. Update `MANIFEST.json` with:
   - dataset,
   - config,
   - seed,
   - best epoch,
   - expected metrics.
3. Compute checksums and update `SHA256SUMS.txt`.

## Why this matters
Even without full training source release, checkpoint + deterministic eval + checksum enables independent verification of reported test metrics.
