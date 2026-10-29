=================
Contributor Guide
=================

Repo structure
==============
``pyproject.toml`` specifies dependencies and tool settings.

``.github/workflows``
    GitHub Actions for formatting, tests and documentation builds.

``.vscode``
    Shared VS Code settings.

``docs``
    Sphinx sources and templates for this site.

``examples``
    Tutorial scripts and generated notebooks. See
    `Documentation and examples`_ for the conversion and publication steps.

``tests``
    Pytest tests, with subfolders matching ``src/mr2``.

``src/mr2/algorithms``
    Data processing and reconstruction algorithms, including k-space
    prewhitening and readout oversampling removal.

``src/mr2/data``
    Containers such as ``KData``, ``IData`` and ``CsmData``. Keep data and
    metadata here; put processing functions in ``algorithms`` unless the
    container needs them to manage its fields.

``src/mr2/operators``
    Linear and nonlinear operators for acquisition models, including
    ``FourierOp`` and ``SensitivityOp``.

``src/mr2/phantoms``
    Numerical phantoms for evaluating reconstruction methods.

``src/mr2/utils``
    Utilities such as spatial filters and ``smap``, which applies a function
    along batch dimensions.


Linting
=======
We use Ruff for linting and `mypy <https://pypi.org/project/mypy/>`_ for
type checking. VS Code recommends the
`Ruff extension <https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff>`_
when you open the repository.

CI runs `pre-commit <https://pre-commit.com/>`_. Install it with
``pip install -e ".[dev]"``, then use ``pre-commit install`` to run hooks
before each commit, or ``pre-commit run --all-files`` to check the repository.

Naming convention
=================
Follow `PEP 8 <https://peps.python.org/pep-0008/>`_: lowercase variable names
and CapWords class names. A source file containing a single class takes
that class's name.

Choose descriptive names such as ``result`` and ``tolerance_squared``.
Use ``n_`` for counts (``n_coils``), ``_op`` for operators (``fourier_op``),
and ``img`` for images. Match nearby code when choosing names and type hints.

Testing
=======
Install the test dependencies with ``pip install -e ".[tests]"`` or the
``dev`` extra, then run pytest. You can also use VS Code's test panel.
Tests must pass before merge. Pytest skips CUDA tests when no CUDA device
is available; use ``pytest -m cuda`` to select them, or
``pytest -m "not cuda"`` to exclude them.

Documentation and examples
==========================
Install ``pip install -e ".[docs]"`` and run ``NORUN=1 make -C docs html``
for a local HTML build without executing notebooks. Open the generated pages
in ``docs/build/html`` and check links, mathematics and code formatting.
Use ``make -C docs clean`` when generated API pages or notebook copies are
stale. A normal build can execute newly copied notebooks; CI runs examples
independently and supplies their executed outputs to the Sphinx build.

Docstrings are the source for IDE help and the API site. ``api.rst`` selects
public modules; recursive autosummary and the templates in
``docs/source/_templates`` generate class and module pages. Public exports
in ``__all__`` control discovery. Use NumPy-style docstrings, documenting
parameters, returns, tensor shapes, units and assumptions. Put public call
semantics on ``__call__`` and computation in ``forward``; users should call
instances so PyTorch runs its module hooks.

Example source lives in ``examples/scripts/*.py``. Mark code cells with
``# %%`` and markdown cells with ``# %% [markdown]``. Pre-commit runs Jupytext
and ``.precommit/add_notebook_preamble.py`` to generate
``examples/notebooks/*.ipynb`` with the Colab badge, install cell and kernel
metadata. Edit the script, then regenerate the notebook.
Commit notebooks with cleared outputs and execution counts; nbstripout
runs before and after conversion and removes ``language_info`` too.

In notebook markdown, single backticks such as `` `~mr2.data.KData` ``
link to Python objects; double backticks mark literal code. Use
``project:other_notebook.ipynb`` for notebook cross-references. Keep ``tags``
and ``mystnb`` cell metadata. A cell tagged ``hide-cell`` with
``mystnb={"code_prompt_show": "Show download details"}`` can fold download,
plotting or callback details while leaving the main reconstruction readable.
State GPU requirements in the tutorial; generated Colab T4 metadata doesn't
establish one.

Pre-commit converts scripts matching ``examples/scripts/*.py``, and Sphinx
copies notebooks from ``examples/notebooks`` into ``docs/source/_notebooks``.
Keep research experiments outside those directories until they are ready
for publication. Add a curated entry in ``examples.rst`` for a new tutorial.
Provide explicit, reproducible downloads from public repositories rather
than committing raw data or other large binary files. State prerequisites,
data size and device needs, and make expensive settings visible to readers.

Code and test design
====================
Prefer small Pythonic changes and existing APIs. Avoid one-off wrappers,
new frameworks for a single use, redundant checks and silent fallback paths.
Keep type hints readable, and document shape and physical assumptions near
the code that depends on them.

Test numerical behavior and shape/broadcast rules. Check adjoint identities
or gradients where the API supports them, and add regression cases for bugs. Choose tolerances appropriate to dtype and algorithm. Prefer a small
input that exposes the error over elaborate fixtures. Avoid duplicate tests
and assertions that merely mirror the implementation. Run the checks relevant
to your change before opening a PR; explain the change and its purpose in
plain technical English.

Release Strategy
================
We are still in pre-release mode and do not guarantee a stable API / strict semantic versioning compatibility. We currently use ``0.YYMMDD`` as versioning and release in regular intervals to `pypi  <https://pypi.org/project/mrtwo/>`_.

Compatibility
=============
We test three environments: the minimum supported Python and dependency versions, the recommended
environment (currently Python 3.12 with PyTorch 2.10), and the latest supported Python with the latest
stable PyTorch release. The recommended environment is also available as a Docker image.
Our type hints will usually only be valid with the latest PyTorch version.
