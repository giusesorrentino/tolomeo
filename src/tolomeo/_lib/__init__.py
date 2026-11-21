"""
Private helpers used by more than one subpackage of tolomeo, with no better place.

The modules here import nothing from the rest of tolomeo, so every module can import
them without circular imports.

* ``messages``: the warnings for the user.
* ``parallel``: the number of threads of a ``workers`` argument.
* ``roma``: the input files of ROMA, a development tool that tolomeo does not use;
  it imports the rest of tolomeo only when it is called.
* ``spectral``: frequency bands of the filters and the preparation of a TOD for a
  spectral estimate.
"""
