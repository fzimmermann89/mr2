.. MRtwo documentation

.. |colab-badge| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/fzimmermann89/mr2

Welcome to MRtwo's documentation!
=================================

MR image reconstruction and processing for PyTorch

| **Source code:** `<https://github.com/fzimmermann89/mr2>`_
| **Bug reports:** `<https://github.com/fzimmermann89/mr2/issues>`_
| **Try it out:** |colab-badge|
| **See our examples:** :doc:`examples`

Start here
----------
New to MRtwo? Follow :doc:`user_guide` to install the package, download an
acquisition and display your first reconstruction. MRtwo is a continuation
of `MRpro <https://github.com/PTB-MR/mrpro/>`_, with a faster-moving development
API. The Python import is ``mr2`` and the PyPI distribution is ``mrtwo``.
This website documents git/main; installed releases can differ.

* **Work with your data:** :doc:`data` explains axes, units, broadcasting,
  indexing and file formats.
* **Build a reconstruction:** :doc:`operators` introduces acquisition models,
  adjoints, losses and solver choices.
* **Learn by application:** :doc:`examples` groups complete notebooks by task.
* **Find an API:** :doc:`api` documents classes, functions and call signatures.
* **Contribute:** :doc:`contributor_guide` describes code, tests and documentation.

MRtwo combines PyTorch tensors with MR metadata, Cartesian and non-Cartesian
Fourier operators, coil sensitivity estimation, regularized reconstruction
and quantitative signal models. Start with a high-level reconstruction and
replace its components as your application requires.

Contents
--------
.. toctree::
   :maxdepth: 2

   user_guide
   data
   operators
   examples
   contributor_guide
   api
   faq
