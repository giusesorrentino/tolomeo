"""
Private helpers used by more than one subpackage of tolomeo, with no better place.

The modules here import nothing from the rest of tolomeo, so every module can import
them without circular imports.

* ``parallel``: the number of threads of a ``workers`` argument.
* ``spectral``: frequency bands of the filters and the preparation of a TOD for a
  spectral estimate.
"""
