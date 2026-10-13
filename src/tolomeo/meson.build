# Top-level package.
#
# Installs the pure Python files of the top-level package and pulls in every
# sub-package. Variables defined in the root meson.build (py, py_dep, np_dep,
# inc_np, inc_f2py, fortranobject_c, quadmath_dep, numpy_nodepr_api,
# conf_data) are visible in all subdirs.
#
# Order matters: if a sub-package depends on a target defined by another one,
# the latter must be listed first.

py.install_sources(
  [
    '__init__.py',
    'version.py',
  ],
  subdir: 'tolomeo'
)

configure_file(
  input: '__config__.py.in',
  output: '__config__.py',
  configuration: conf_data,
  install_dir: py.get_install_dir() / 'tolomeo',
  install_tag: 'python-runtime',
)
