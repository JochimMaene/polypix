# Architecture decisions

Architecture decision records explain why Polypix owns a focused native RING
kernel and why a small number of fused reductions were admitted. They are
maintainer evidence rather than user instructions:

- [Owned HEALPix kernel](https://github.com/JochimMaene/polypix/blob/main/decisions/owned-healpix-kernel.md)
- [Exact caps and segmented occupancy](https://github.com/JochimMaene/polypix/blob/main/decisions/cap-and-occupancy-primitives.md)
- [API surface beyond constellation examples](https://github.com/JochimMaene/polypix/blob/main/decisions/api-surface-beyond-constellations.md)
- [Coverage reductions and ordinal occupancy runs](https://github.com/JochimMaene/polypix/blob/main/decisions/coverage-reductions-and-occupancy-runs.md)

For what Polypix actually supports, see the [user guide](concepts.md) and the
[API reference](api.md).
