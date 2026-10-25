# Third-party notices

ImageProcessingFlex is licensed under the MIT License (see `LICENSE`). It contains no
third-party source code, but it includes the following data and is based in part on the
following works, which are credited here as their licenses require or request.

## Colormap tables (`include/ipf/colormap_data.hpp`)

- **Viridis, Inferno, Magma, Plasma, Cividis** — from matplotlib
  (`lib/matplotlib/_cm_listed.py`), created by Stéfan van der Walt, Nathaniel Smith,
  Eric Firing (Viridis/Magma/Inferno/Plasma) and Jamie Nuñez, Ryan Renslow, Christopher Anderton
  (Cividis). Released under the CC0 1.0 Universal public domain dedication.
- **Turbo** — Copyright 2019 Google LLC, created by Anton Mikhailov. Licensed under the
  Apache License, Version 2.0 (https://www.apache.org/licenses/LICENSE-2.0). The 256-entry table
  is included as 8-bit values (rounded from the original floating-point data).

## JPEG decoder (`include/ipf/codec/jpeg.hpp`)

The JPEG decoder implements ITU-T Recommendation T.81 (including the probability estimation
table D.2 of the arithmetic coder). Its progressive refinement and arithmetic decoding
procedures follow the structure of the libjpeg / libjpeg-turbo decoders.

This software is based in part on the work of the Independent JPEG Group.

## Test data (`tests/data`, not part of the library headers)

- `test3.icc` is a test profile from libjpeg-turbo's `testimages` directory.
- `*.jpg` / `*.pnm` reference files were produced with libjpeg-turbo's `cjpeg` and `djpeg`.

libjpeg-turbo is covered by the IJG License and the Modified (3-clause) BSD License
(https://github.com/libjpeg-turbo/libjpeg-turbo/blob/main/LICENSE.md).
