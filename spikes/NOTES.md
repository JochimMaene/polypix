# Direct RING Kernel Prototype

## Question

Can a Polypix-owned, center-only HEALPix kernel remove enough work from the
CDS/BMOC path to make a 10x end-to-end product improvement credible? The
stretch target was a 15x native margin so Python input handling would not erase
the gain.

## Verdict

**Proceed with an owned RING-first kernel, but do not promote this spike as-is.**

The prototype returns exactly the same center memberships as the current
CDS-backed NESTED kernel on the standard scorecard. A separate differential
run also passed 198 globally distributed quadrilaterals at five resolutions in
serial and automatic thread modes, including exact poles, longitude seams,
reversed winding, and non-unit input vectors.

One representative 21-repeat single-thread run on the Intel i7-1165G7 host:

| Workload | Current CDS | Prototype RING | Speedup |
| --- | ---: | ---: | ---: |
| 4,096 dense quads, resolution 6 | 19.328 ms | 1.787 ms | 10.82x |
| 4,096 dense quads, resolution 9 | 152.268 ms | 9.120 ms | 16.70x |
| 2,048 ragged 3--6 vertex polygons, resolution 9 | 93.226 ms | 12.349 ms | 7.55x |
| 4,096 strip quads, resolution 9 | 72.952 ms | 4.278 ms | 17.05x |
| One footprint, resolution 9 | 0.056 ms | 0.009 ms | 6.15x |

Automatic parallel measurements are directionally stronger for dense
high-resolution output but noisier on this host. They should not replace a
proper multi-host benchmark campaign.

The 15x stretch target is not universal. The 10x feasibility question is
answered positively for the primary fixed-quadrilateral and strip batches.
Ragged batches and single-footprint latency need more work; no broader claim is
justified yet.

## Why It Is Faster

The current implementation computes exact polygon/cell overlap through a BMOC
and then filters partial cells by center. The prototype instead:

1. uses the `4 * nside - 1` HEALPix iso-latitude rings directly;
2. bounds a validated quadrilateral once in latitude and longitude;
3. walks only the possible RING centers;
4. applies four unrolled half-space tests;
5. appends standard RING IDs directly;
6. prepares fixed quadrilaterals and output buffers without per-polygon heap
   allocation;
7. parallelizes coarse, independent input chunks.

There is no recursive cell traversal, overlap geometry, intermediate MOC,
sorting pass, or backend abstraction.

## Architectural Answer

RING is not merely an internal curiosity here. It is the natural output of the
winning algorithm. Converting every emitted ID to NESTED costs a material share
of high-output calls and would retain mapping code that the direct kernel does
not otherwise need. Because breaking changes are acceptable in `0.x`, the
target contract should become RING-only unless downstream compatibility
evidence outweighs the measured cost.

The fast path is intentionally narrow: dense quadrilaterals. That matches the
main satellite-footprint and strip workload. Production work should add a
native paired-edge entry point so Python never materializes strip quads, and
should use the same fixed-capacity approach for 3--6 vertex ragged polygons
before claiming a library-wide 10x improvement.

Sparse candidate cells were not part of this prototype. The next bounded
experiment is to convert candidates to RING once, sort them once, and intersect
them with emitted ring spans. Keep it only if it beats direct half-space tests.

## Promotion Gates

Before replacing CDS:

- compare against independent brute-force center enumeration at tractable
  resolutions and a large fixed-seed adversarial corpus;
- prove ring numbering and centers through resolution 29 at polar and
  equatorial transitions;
- implement the native strip input path and optimize the documented ragged
  input path;
- benchmark complete public calls, including Python conversion and result
  construction, on x86-64 and ARM64;
- measure automatic-thread policy, peak memory, wheel size, and source-build
  dependency reduction;
- remove the benchmark-only Python entry point and move only the small kernel
  into the production module.

Run the reproducible prototype with:

```bash
pixi run -e test ring-prototype --profile standard --repeats 21 --threads 1
```
