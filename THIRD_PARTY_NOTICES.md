# Third-Party Notices

Polypix links against HEALPix C++ for HEALPix geometry operations. Binary wheels
may also include native runtime libraries needed by HEALPix C++ and CFITSIO.

The Polypix distribution is licensed under GPL-3.0-or-later. HEALPix is
available under GPL-2.0-or-later, which permits redistribution under GPL-3.0 or
later. The wheel build uses the GPL-3.0-or-later option for the combined binary
distribution.

## Direct Native Dependencies

| Component | License | Source |
| --- | --- | --- |
| HEALPix C++ | GPL-2.0-or-later | https://healpix.sourceforge.io |
| CFITSIO | LicenseRef-fitsio | https://heasarc.gsfc.nasa.gov/fitsio/ |

## Runtime Libraries Bundled By Wheel Repair

Depending on platform and dependency resolution, repaired wheels may bundle
runtime libraries from this set:

| Component | License |
| --- | --- |
| bzip2 | bzip2-1.0.6 |
| curl / libcurl | curl |
| GCC runtime libraries | GPL-3.0-only WITH GCC-exception-3.1 |
| Kerberos libraries | MIT and LGPL-2.1-or-later components |
| libnghttp2 | MIT |
| libssh2 | BSD-3-Clause |
| OpenSSL | Apache-2.0 |
| zlib | Zlib |
| zstd | BSD-3-Clause |

Release builds should include the corresponding license texts for all bundled
native runtime libraries in the wheel. The exact bundled set is determined by
`auditwheel`, `delocate`, or `delvewheel` during wheel repair.

## Source Availability

For GPL compliance, source archives for GPL-covered bundled components must be
available for the corresponding binary wheel release. Release automation should
record the exact dependency versions and source URLs used for each wheel build.
