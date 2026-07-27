# Third-Party Notices

Polypix includes a native Rust kernel. Rust crate dependencies are compiled into
the extension module distributed in binary wheels; users do not need to install
them separately.

## Direct Native Dependencies

| Component | License used by Polypix | Source |
| --- | --- | --- |
| CDS HEALPix (`cdshealpix`) | Apache-2.0, selected from its `MIT OR Apache-2.0` options | https://github.com/cds-astro/cds-healpix-rust |
| PyO3 | Apache-2.0, selected from its `MIT OR Apache-2.0` options | https://github.com/PyO3/pyo3 |
| rust-numpy | BSD-2-Clause | https://github.com/PyO3/rust-numpy |
| Rayon | Apache-2.0, selected from its `MIT OR Apache-2.0` options | https://github.com/rayon-rs/rayon |

The exact versions and complete transitive dependency graph for a release are
recorded in `Cargo.lock`. Every wheel also contains Maturin's generated
CycloneDX SBOM, which records the complete resolved Rust graph with versions and
SPDX license expressions.

The locked graph was audited as a whole. Where a dependency offers
Apache-2.0 as an option, Polypix uses that option and the license text in
`LICENSE` applies. The remaining dependencies and required additional terms
are:

| Components | Terms used by Polypix |
| --- | --- |
| `bincode`, `memoffset`, `simd-adler32`, `slab`, `winnow` | MIT |
| `byteorder`, `katex-doc`, `memchr` | Unlicense |
| rust-numpy (`numpy`) | BSD-2-Clause |
| `unicode-ident` | Apache-2.0 AND Unicode-3.0 |
| `target-lexicon` | Apache-2.0 WITH LLVM-exception |
| `colorous` incorporated color data | Upstream NOTICE terms reproduced below |

The corresponding notices are reproduced below. This file and the generated
SBOM must remain in binary distributions. Changes to `Cargo.lock` require this
audit to be repeated before release.

## MIT Notices

The following upstream copyright notices apply:

- `bincode`: Copyright (c) 2014 Ty Overby
- `memoffset`: Copyright (c) 2017 Gilad Naaman
- `simd-adler32`: Copyright (c) 2021 Marvin Countryman
- `slab`: Copyright (c) 2019 Carl Lerche

The upstream `winnow` MIT license contains no separate copyright line.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## BSD-2-Clause Notice For rust-numpy

Copyright (c) 2017, Toshiki Teramura
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

- Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Unlicense Notice

This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or distribute this
software, either in source code form or as a compiled binary, for any purpose,
commercial or non-commercial, and by any means.

In jurisdictions that recognize copyright laws, the author or authors of this
software dedicate any and all copyright interest in the software to the public
domain. We make this dedication for the benefit of the public at large and to
the detriment of our heirs and successors. We intend this dedication to be an
overt act of relinquishment in perpetuity of all present and future rights to
this software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

For more information, see <http://unlicense.org/>.

## Unicode-3.0 Notice

Copyright © 1991-2023 Unicode, Inc.

NOTICE TO USER: Carefully read the following legal agreement. BY DOWNLOADING,
INSTALLING, COPYING OR OTHERWISE USING DATA FILES, AND/OR SOFTWARE, YOU
UNEQUIVOCALLY ACCEPT, AND AGREE TO BE BOUND BY, ALL OF THE TERMS AND CONDITIONS
OF THIS AGREEMENT. IF YOU DO NOT AGREE, DO NOT DOWNLOAD, INSTALL, COPY,
DISTRIBUTE OR USE THE DATA FILES OR SOFTWARE.

Permission is hereby granted, free of charge, to any person obtaining a copy of
data files and any associated documentation (the "Data Files") or software and
any associated documentation (the "Software") to deal in the Data Files or
Software without restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, and/or sell copies of the Data Files
or Software, and to permit persons to whom the Data Files or Software are
furnished to do so, provided that either (a) this copyright and permission
notice appear with all copies of the Data Files or Software, or (b) this
copyright and permission notice appear in associated Documentation.

THE DATA FILES AND SOFTWARE ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT OF THIRD
PARTY RIGHTS.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR HOLDERS INCLUDED IN THIS NOTICE BE
LIABLE FOR ANY CLAIM, OR ANY SPECIAL INDIRECT OR CONSEQUENTIAL DAMAGES, OR ANY
DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN
CONNECTION WITH THE USE OR PERFORMANCE OF THE DATA FILES OR SOFTWARE.

Except as contained in this notice, the name of a copyright holder shall not be
used in advertising or otherwise to promote the sale, use or other dealings in
these Data Files or Software without prior written authorization of the
copyright holder.

## LLVM Exception

As an exception, if, as a result of your compiling your source code, portions
of this Software are embedded into an Object form of such source code, you may
redistribute such embedded portions in such Object form without complying with
the conditions of Sections 4(a), 4(b) and 4(d) of the Apache License 2.0.

In addition, if you combine or link compiled forms of this Software with
software that is licensed under the GPLv2 ("Combined Software") and if a court
of competent jurisdiction determines that the patent provision (Section 3),
the indemnity provision (Section 9) or other Section of the License conflicts
with the conditions of the GPLv2, you may retroactively and prospectively
choose to deem waived or otherwise exclude such Section(s) of the License, but
only in their entirety and only with respect to the Combined Software.

## colorous NOTICE

This project incorporates some elements from d3-scale-chromatic, which carries
the following copyright notice:

<https://github.com/d3/d3-scale-chromatic>

Copyright 2010-2018 Mike Bostock
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

- Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
- Neither the name of the author nor the names of contributors may be used to
  endorse or promote products derived from this software without specific
  prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Apache-Style Software License for ColorBrewer software and ColorBrewer Color
Schemes

Copyright (c) 2002 Cynthia Brewer, Mark Harrower, and The Pennsylvania State
University.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at

<http://www.apache.org/licenses/LICENSE-2.0>

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

Polypix no longer links to HEALPix C++ or CFITSIO. Those dependencies and their
GPL distribution requirements apply to earlier Polypix releases only.
