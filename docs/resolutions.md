# Resolutions

One number controls the whole grid. `resolution` is the HEALPix order, and
everything else follows from it:

```text
nside      = 2 ** resolution
cell_count = 12 * 4 ** resolution
```

Polypix accepts 0 through 29.

## Reading the table

HEALPix is an equal-area grid, so every cell on the sphere covers exactly the
same solid angle. Their *shapes* differ — cells near the poles are not the same
shape as cells on the equator — so there is no single edge length that holds
everywhere. The nominal cell size below is `sqrt(cell area)`, which is the usual
stand-in and matches what `healpy.nside2resol()` reports. Treat it as a typical
size, not as a bound on how wide any particular cell gets.

The Earth column is that same angle laid on a sphere of radius 6,371 km, the
IUGG mean radius. It is the distance across a typical cell at ground level.

{.polypix-reftable}
| Resolution | `nside` | Cells | Nominal cell size | On Earth | Dense `int64` map |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 12 | 58.63° | 6,520 km | 96 B |
| 1 | 2 | 48 | 29.32° | 3,260 km | 384 B |
| 2 | 4 | 192 | 14.66° | 1,630 km | 1.5 KiB |
| 3 | 8 | 768 | 7.329° | 815 km | 6 KiB |
| 4 | 16 | 3,072 | 3.665° | 407.5 km | 24 KiB |
| 5 | 32 | 12,288 | 1.832° | 203.7 km | 96 KiB |
| 6 | 64 | 49,152 | 54.97′ | 101.9 km | 384 KiB |
| 7 | 128 | 196,608 | 27.48′ | 50.93 km | 1.5 MiB |
| 8 | 256 | 786,432 | 13.74′ | 25.47 km | 6 MiB |
| 9 | 512 | 3,145,728 | 6.871′ | 12.73 km | 24 MiB |
| 10 | 1,024 | 12,582,912 | 3.435′ | 6.367 km | 96 MiB |
| 11 | 2,048 | 50,331,648 | 1.718′ | 3.183 km | 384 MiB |
| 12 | 4,096 | 201,326,592 | 51.53″ | 1.592 km | 1.5 GiB |
| 13 | 8,192 | 805,306,368 | 25.77″ | 795.9 m | 6 GiB |
| 14 | 16,384 | 3,221,225,472 | 12.88″ | 397.9 m | 24 GiB |
| 15 | 32,768 | 12,884,901,888 | 6.442″ | 199 m | 96 GiB |
| 16 | 65,536 | 51,539,607,552 | 3.221″ | 99.48 m | 384 GiB |
| 17 | 131,072 | 206,158,430,208 | 1.61″ | 49.74 m | 1.5 TiB |
| 18 | 262,144 | 824,633,720,832 | 805.2 mas | 24.87 m | 6 TiB |
| 19 | 524,288 | 3,298,534,883,328 | 402.6 mas | 12.44 m | 24 TiB |
| 20 | 1,048,576 | 13,194,139,533,312 | 201.3 mas | 6.218 m | 96 TiB |
| 21 | 2,097,152 | 52,776,558,133,248 | 100.6 mas | 3.109 m | 384 TiB |
| 22 | 4,194,304 | 211,106,232,532,992 | 50.32 mas | 1.554 m | 1.5 PiB |
| 23 | 8,388,608 | 844,424,930,131,968 | 25.16 mas | 777.2 mm | 6 PiB |
| 24 | 16,777,216 | 3,377,699,720,527,872 | 12.58 mas | 388.6 mm | 24 PiB |
| 25 | 33,554,432 | 13,510,798,882,111,488 | 6.291 mas | 194.3 mm | 96 PiB |
| 26 | 67,108,864 | 54,043,195,528,445,952 | 3.145 mas | 97.15 mm | 384 PiB |
| 27 | 134,217,728 | 216,172,782,113,783,808 | 1.573 mas | 48.57 mm | 1.5 EiB |
| 28 | 268,435,456 | 864,691,128,455,135,232 | 786.3 µas | 24.29 mm | 6 EiB |
| 29 | 536,870,912 | 3,458,764,513,820,540,928 | 393.2 µas | 12.14 mm | 24 EiB |

## Picking one

Match the grid to the smallest feature you care about, then stop. Each step up
quadruples the cell count and halves the cell size, so overshooting by two costs
you sixteen times the memory for detail you are not using.

Some rough anchors:

- **Resolution 6** (≈100 km) suits constellation coverage and revisit studies.
  Both [case studies](examples/index.md) use it.
- **Resolution 8–10** (25 km down to 6 km) suits regional analysis and most
  sensor-footprint work.
- **Resolution 12** (1.6 km) is about where a dense global map stops being
  comfortable — one `int64` per cell is already 1.5 GiB.
- **Above 13**, forget dense global arrays. The high resolutions exist for
  sparse work: pass `candidate_cells=` to restrict coverage to cells you care
  about, or `cells=` to query a short list. See
  [Performance and memory](performance.md).

The last few rows are there for completeness rather than use. Resolution 29
divides the Earth into 12 millimetre cells and would need 24 EiB to hold one
integer each.
