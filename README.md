<h1 align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/fzimmermann89/mr2/refs/heads/main/docs/source/_static/logo_white.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/fzimmermann89/mr2/refs/heads/main/docs/source/_static/logo.svg">
  <img src="https://raw.githubusercontent.com/fzimmermann89/mr2/refs/heads/main/docs/source/_static/logo.svg" alt="MRtwo logo" width="50%">
</picture>

</h1><br>

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://pypi.org/project/mrtwo/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Coverage Bagde](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/fzimmermann89/f688ae5c6e8daec44ac7f8fc4067e93f/raw/coverage.json)](https://github.com/fzimmermann89/mr2/actions?query=workflow%3A%22%22Report+PyTest%22%22)
[![DOI](https://img.shields.io/badge/DOI-10.1002%2Fnbm.70405-blue)](https://doi.org/10.1002/nbm.70405)
[![FAIR checklist badge](https://fairsoftwarechecklist.net/badge.svg)](https://fairsoftwarechecklist.net/v0.2?f=31&a=32113&i=21322&r=133)

MR data processing and image reconstruction.

This project is a faster moving continuation/fork of [MRpro](https://github.com/PTB-MR/mrpro/).
Some of the additional features of MRtwo will eventually get backported to MRpro, all new features of MRpro will be included in MRtwo.

In most cases, you can replace `mrpro` by `mr2` in your code and everything works.

- **Source code:** <https://github.com/fzimmermann89/mr2>
- **Documentation:** <https://fzimmermann89.github.io/mr2/>
- **Bug reports:** <https://github.com/fzimmermann89/mr2/issues>
- **Try it out:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fzimmermann89/mr2)

## Main features

- Data containers store MR metadata with PyTorch tensors and support indexing
  and rearranging fields together.
- Cartesian and non-Cartesian Fourier operators support reconstruction with
  coil sensitivity maps, density compensation and regularization.
- Signal models support quantitative imaging, including T1 recovery and WASABI.
- Neural network components support learned MRI reconstruction.
- Raw-data loaders read ISMRMRD; trajectory calculators can use stored
  coordinates or a [Pulseq](http://pulseq.github.io/) sequence file.

## Quickstart

Install the PyPI distribution, then import `mr2`:

```bash
pip install mrtwo
```

```python
import mr2

# Use your ISMRMRD file with its stored trajectory.
trajectory = mr2.data.traj_calculators.KTrajectoryIsmrmrd()
kdata = mr2.data.KData.from_file('acquisition.h5', trajectory)
reconstruction = mr2.algorithms.reconstruction.DirectReconstruction(kdata)
img = reconstruction(kdata)  # IData: complex image tensor and image header
```

The [first reconstruction guide](https://fzimmermann89.github.io/mr2/user_guide.html)
includes a complete download-and-plot example. Continue with
[data containers](https://fzimmermann89.github.io/mr2/data.html),
[operators](https://fzimmermann89.github.io/mr2/operators.html), or the
[grouped examples](https://fzimmermann89.github.io/mr2/examples.html).
Install `mrtwo[notebooks]` for example dependencies. The website follows
`main`; features added since the latest PyPI release require a
[development install](https://fzimmermann89.github.io/mr2/user_guide.html#installation).

## Development

`pip install -e ".[dev]"`

MRtwo requires PyTorch 2.3.1 or newer. CI currently validates the minimum
supported CPU build, PyTorch 2.3.1, and the recommended CPU build, PyTorch 2.10.0.

### Docker

A notebook-capable image for the recommended environment (Python 3.12 and PyTorch 2.10) is available from GHCR:

```bash
docker pull ghcr.io/fzimmermann89/mr2_py312:latest
```

## Citation

MRtwo is a continuation of MRpro. If you use MRtwo, please cite the MRpro framework paper:

F. F. Zimmermann, P. Schuenke, C. S. Aigner, B. A. Bernhardt, M. Guastini, J. Hammacher,
H. Herthum, N. Jaitner, A. Kofler, L. Lunin, S. Martin, C. Redshaw Kranich, J. Schattenfroh,
D. Schote, Y. Wu, and C. Kolbitsch (2026).
**MRpro: Open Framework for Model-Based, Learned, and Quantitative MR Imaging—Application to Low-Field MRI.**
*NMR in Biomedicine*, 39(11), e70405. Zimmermann and Schuenke contributed equally.
[doi:10.1002/nbm.70405](https://doi.org/10.1002/nbm.70405)
