=================
Contributor Guide
=================

Repo structure
==============
This repository uses a *pyproject.toml* file to specify all the requirements.

**.github/workflows**
    Definitions of GitHub action workflows to carry out formatting checks, run tests and automatically create this
    documentation.

**.vscode**
    Configuration files for VS Code.

**docs**
    Files to create this documentation.

**examples**
    Python scripts showcasing how MRtwo can be used. Any data needed has to be available from
    an online repository (e.g. zenodo) such that it can be automatically downloaded.
    Individual cells should be indicated with ``# %%``. For markdown cells use ``# %% [markdown]``.
    The translation from Python script to Jupyter notebook is done in pre-commit (locally and on GitHub)
    using `jupytext <https://jupytext.readthedocs.io/en/latest/>`_ . See its documentation for more details.

    After translating the scripts to notebooks, the notebooks are run and their output is converted to html and added
    to this documentation in the *Examples* section.

    All output cells in the notebooks are automatically cleared, and only cleared notebooks should be added to the repository.

**tests**
    Tests which are automatically run by pytest.
    The subfolder structure should follow the same structure as in *src/mr2*.

**src/mr2/algorithms**
    Everything which does something with the data, e.g. prewhiten k-space or remove oversampling.

**src/mr2/data**
    All the data classes such as ``KData``, ``IData`` or ``CsmData``.
    As the names suggest these should mainly contain data and meta information.
    Any functionality beyond what is absolutely required for the classes should be put as separate functions.

**src/mr2/operators**
    Linear and non-linear algorithms describing e.g. the transformation from image to k-space (``FourierOp``), the
    effect of receiver coils (``SensitivityOp``) or MR signal models.

**src/mr2/phantoms**
    Numerical phantoms useful to evaluate reconstruction algorithms.

**src/mr2/utils**
    Utilities such as spatial filters and also more basic functionality such as applying functions serially along the
    batch dimension (``smap``).


Linting
=======
We use Ruff for linting. If you are using VSCode, you can install an
`extension <https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff>`_,
which we have also added to the list of extensions that VSCode should recommend when you open the code.
We also run `mypy <https://pypi.org/project/mypy/>`_ as a type checker.

In CI, our linting is driven by `pre-commit <https://pre-commit.com/>`_.
If you install MRtwo via ``pip install -e .[dev]``, pre-commit will be installed in your Python environment.
You can either add pre-commit to your git pre-commit hooks, requiring it to pass before each commit (``pre-commit install``),
or run it manually using ``pre-commit run --all-files`` after making your changes, before requesting a PR review.

Naming convention
=================
We try to follow the `PEP 8 <https://peps.python.org/pep-0008/>`_ naming convention (e.g., all lowercase variable names,
CapWords class names). We deviate for the names of source code file names containing a single class.
These are named as the class.

We try to use descriptive variable names when applicable (e.g., ``result`` instead of ``res``, ``tolerance_squared`` instead
of ``sqtol``, ``batchsize`` instead of ``m``).

A name starting with ``n_`` is used for variables describing a number of something (e.g., ``n_coils`` instead of ``ncoils`` or
``num_coils``), variable names ending with ``_op`` for operators (e.g., ``fourier_op``). We use ``img`` as a variable name
for images.

Testing
=======
We use pytest for testing. All required packages will be installed if you install MRtwo via ``pip install -e .[dev]`` or ``pip install -e .[tests]``.
You can use VSCode's test panel to discover and run tests. All tests must pass before a PR can be merged.
CUDA tests run when a CUDA device is available and are skipped otherwise. Use ``pytest -m cuda`` to run only CUDA tests,
or ``pytest -m "not cuda"`` to exclude them.

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
instances so PyTorch hooks are respected.

Example source lives in ``examples/scripts/*.py``. Mark code cells with
``# %%`` and markdown cells with ``# %% [markdown]``. Pre-commit runs Jupytext
and ``.precommit/add_notebook_preamble.py`` to generate
``examples/notebooks/*.ipynb`` with the Colab badge, install cell and kernel
metadata. Do not maintain a second version by editing the notebook directly.
Only commit notebooks with outputs and execution counts cleared; nbstripout
runs before and after conversion and removes ``language_info`` too.

In notebook markdown, single backticks such as `` `~mr2.data.KData` ``
link to Python objects; double backticks mark literal code. Use
``project:other_notebook.ipynb`` for notebook cross-references. Keep ``tags``
and ``mystnb`` cell metadata. A cell tagged ``hide-cell`` with
``mystnb={"code_prompt_show": "Show download details"}`` can fold download,
plotting or callback details while leaving the main reconstruction readable.
The generated Colab T4 metadata does not establish a GPU requirement.

Scripts matching the pre-commit pattern are converted broadly, and Sphinx
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
Use readable type hints and descriptive names consistent with nearby code:
``n_coils`` for counts, ``img`` for images and ``fourier_op`` for operators.
Keep shape and physical assumptions near the code that depends on them.

Tests should establish behavior: numerical results, adjoint identities,
gradients where supported, shape/broadcast rules and regressions for actual
bugs. Choose tolerances appropriate to dtype and algorithm. Prefer a small
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
