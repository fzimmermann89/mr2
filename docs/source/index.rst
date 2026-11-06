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
Follow :doc:`user_guide` to install MRtwo and reconstruct a downloaded
acquisition. MRtwo continues development of
`MRpro <https://github.com/PTB-MR/mrpro/>`_. Install the ``mrtwo`` distribution
and import ``mr2`` in Python. This website documents git/main, so some APIs
may differ from the latest PyPI release.

* :doc:`data` explains tensor axes and units, with examples of indexing and
  file storage.
* :doc:`operators` covers acquisition models, operator algebra and solver
  selection.
* :doc:`examples` groups complete notebooks by application.
* :doc:`api` lists public classes and functions with their call signatures.
* :doc:`contributor_guide` describes the code and documentation conventions.

MRtwo stores MR metadata alongside PyTorch tensors. Its reconstruction tools
include Cartesian and non-Cartesian Fourier operators, coil sensitivity
estimation and regularized solvers; signal models support quantitative
parameter fitting. Use a high-level reconstruction first, then replace its
components when you need a different model or solver.

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
