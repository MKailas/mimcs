API reference
=============

Generated from the docstrings. Documented **per package** rather than per module: only the
package ``__init__`` files carry ``__all__``, so a package gives a curated surface while a leaf
module would pull in every re-imported symbol.

One thing autodoc cannot show: the sampler class you get from :func:`mimcs.make_sampler` is built
at runtime by :func:`~mimcs.make_sampler_class`, which composes mixins with ``type()``. Those
classes have no source to document --- see ``design/02_sampler_classes.md`` for how they work.

.. toctree::
   :maxdepth: 1

   mimcs
   mimcs_model
   mimcs_dsl
   mimcs_factory
   mimcs_samplers
   mimcs_hmc
   mimcs_adaptation
   mimcs_pt
   mimcs_rng
   mimcs_diagnostics
   mimcs_summary
   mimcs_optim
   mimcs_testing
