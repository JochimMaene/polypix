# Recorded benchmark matrix

Collected on an Intel Core i7-1165G7 under WSL2. These are development-machine
measurements, not an isolated performance laboratory. Each JSON preserves raw
cold and warmed timings, process peak RSS, accuracy metrics, and environment.
The source revision identifies the library checkout; the benchmark was added as
uncommitted work. `polypix-build.json` identifies the actual local native module,
whose version is absent from this temporary environment's package metadata.

`environment.txt` freezes the optional competitor environment. The Polypix
extension was loaded from the checkout. Chart code uses Matplotlib from the docs
environment. See `../competitors.md` for the full reproduction commands.

Reports retain the harness hash as executed. A subsequent import-order-only lint
fix changes that hash without changing the measurement code.
