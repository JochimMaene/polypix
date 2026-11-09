# Benchmarks

How does Polypix compare with the other libraries you might reach for? We ran it
next to healpy and HPGeom, which use the same HEALPix grid, and next to MOCPy, H3,
and S2, which cover shapes on their own grids. In short: on one thread, Polypix
is 4 to 6 times faster than healpy at counting overlapping circles, and 20 to 40
times faster at covering many small polygons.

All numbers come from one laptop (Intel i7-1165G7, Linux under WSL2), so treat
the ratios as a guide rather than a promise.

## Two workloads

**Visibility** counts, for every cell, how many circles cover it. Picture a
constellation where each satellite serves a circle on the ground: visibility
tells you how many satellites each place can see.

**Observation** finds every cell covered by at least one footprint. Picture an
imaging satellite taking overlapping pictures along its track: observation
tells you which parts of the Earth it has seen.

The circles have radii between 3° and 12°. The footprints are 1.4° by 0.9°,
four of them in a row per track. Every library gets exactly the same shapes.

## Against other HEALPix libraries

healpy and HPGeom select exactly the same cells as Polypix, which the benchmark
checks every time, so this is a pure speed comparison. Here we use more shapes:
10,000 circles and 4,000 footprints.

```{figure} assets/benchmarks/visibility-healpix.svg
:class: only-light
:alt: Bar chart of milliseconds per call for 10,000 circles at HEALPix orders 5, 6 and 7. Polypix takes 4.6, 8.2 and 14.5 ms on one thread, healpy 26, 34 and 58 ms, HPGeom 90, 97 and 122 ms.
```

```{figure} assets/benchmarks/visibility-healpix-dark.svg
:class: only-dark
:alt: Bar chart of milliseconds per call for 10,000 circles at HEALPix orders 5, 6 and 7. Polypix takes 4.6, 8.2 and 14.5 ms on one thread, healpy 26, 34 and 58 ms, HPGeom 90, 97 and 122 ms.
```

```{figure} assets/benchmarks/observation-healpix.svg
:class: only-light
:alt: Bar chart of milliseconds per call for 4,000 footprints at HEALPix orders 5, 6 and 7. Polypix takes 2.3 to 3.3 ms on one thread, healpy 46 to 126 ms, HPGeom 69 to 151 ms.
```

```{figure} assets/benchmarks/observation-healpix-dark.svg
:class: only-dark
:alt: Bar chart of milliseconds per call for 4,000 footprints at HEALPix orders 5, 6 and 7. Polypix takes 2.3 to 3.3 ms on one thread, healpy 46 to 126 ms, HPGeom 69 to 151 ms.
```

healpy and HPGeom cover one shape per Python call, so their time grows with the
number of shapes. Polypix takes the whole batch in one call. That gap is largest
for many small shapes, which is why observation shows the bigger difference.
With threads, Polypix gets another 1.5 to 2.5 times faster on these inputs.

## Against other grids

H3 and S2 don't use HEALPix, and their cells come in different sizes: HEALPix
has 12 × 4ⁿ cells, S2 has 6 × 4ⁿ, and H3 grows sevenfold per level. So we run
each library at several of its own levels and plot the time against the number
of cells on the globe. These runs use 128 circles and 512 footprints, because
H3 and S2 become slow with more.

Libraries also disagree on when a cell counts as covered. Some take a cell when
its center is inside the shape, others when any part of it touches the shape.
Polypix supports both, so the chart has one column per rule, and only libraries
that use the same rule share a column.

```{figure} assets/benchmarks/grids.svg
:class: only-light
:alt: Milliseconds per call against cell count, for circles and footprints, with the center rule and the touch rule. Center rule, circles: Polypix 0.04 to 3.3 ms, healpy 0.3 to 5 ms, HPGeom 1.1 to 6.3 ms, H3 39 to 219 ms. Center rule, footprints: Polypix 0.25 to 2.4 ms, the others 4 to 37 ms. Touch rule, circles: Polypix 2.6 to 39 ms, MOCPy 2.9 to 45 ms, S2 4.9 to 183 ms, H3 49 to 275 ms. Touch rule, footprints: MOCPy 3.4 to 11 ms, Polypix 8 to 14 ms, S2 11 to 23 ms, H3 51 to 105 ms.
```

```{figure} assets/benchmarks/grids-dark.svg
:class: only-dark
:alt: Milliseconds per call against cell count, for circles and footprints, with the center rule and the touch rule. Center rule, circles: Polypix 0.04 to 3.3 ms, healpy 0.3 to 5 ms, HPGeom 1.1 to 6.3 ms, H3 39 to 219 ms. Center rule, footprints: Polypix 0.25 to 2.4 ms, the others 4 to 37 ms. Touch rule, circles: Polypix 2.6 to 39 ms, MOCPy 2.9 to 45 ms, S2 4.9 to 183 ms, H3 49 to 275 ms. Touch rule, footprints: MOCPy 3.4 to 11 ms, Polypix 8 to 14 ms, S2 11 to 23 ms, H3 51 to 105 ms.
```

What stands out:

- **With the center rule, Polypix is the fastest everywhere.** For footprints
  it is 14 to 30 times faster than healpy. With only 128 circles the gap to
  healpy shrinks to 1.5 to 7 times, while H3 is a few hundred times slower.
- **For small polygons under the touch rule, MOCPy is faster than Polypix.**
  Polypix spends around 8 ms here even on the coarsest grid, so this is an area
  where it has room to improve.

## What about accuracy?

Any grid gets the edges of a shape slightly wrong: a cell that is partly inside
has to be either in or out. That error comes from using cells at all, and it
shrinks as the cells get smaller. We checked it at 100,000 random points, and at
the same number of cells HEALPix, H3, and S2 all land within about 20% of each
other. So pick a grid for its other strengths, and pick the resolution for the
accuracy you need; the [resolution guide](resolutions.md) helps with that.

The one difference between libraries is under the touch rule. MOCPy
approximates circles, so it adds 9 to 15% more edge cells than Polypix, which
tests the exact boundary.

## How we measured

Each case runs in five fresh processes. Each process makes one warm-up call and
five timed calls, and we report the median. Times cover the coverage call and
turning its output into per-cell counts; they exclude imports and creating the
shapes. Polypix runs on one thread unless the label says threaded, and MOCPy is
held to one thread. healpy, HPGeom, H3, and S2 are single-threaded.

H3 has no circle primitive, so it gets a 128-point outline of each circle, and
building that outline counts towards its time. It also draws polygon edges as
straight lines in latitude and longitude rather than great circles; the edges
here are short enough that this makes no visible difference.

S2's Python bindings return one object per cell, and turning those into integer
IDs is about half of S2's time on the finest grid. That's a real cost if you
use S2 from Python, but the C++ library itself is faster than these numbers.

To run the benchmark yourself, see
[benchmarks/competitors.md](https://github.com/JochimMaene/polypix/blob/main/benchmarks/competitors.md).
