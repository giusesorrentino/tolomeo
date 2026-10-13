project(
  'tolomeo',
  # 'c' is required because f2py generates a C wrapper and fortranobject.c
  ['fortran', 'c'],
  license: 'MPL-2.0',
  meson_version: '>= 1.9.0',
  version: '0.1.0',
  default_options: ['buildtype=release']
)

fc = meson.get_compiler('fortran')
cc = meson.get_compiler('c')

py = import('python').find_installation(pure: false)
py_dep = py.dependency()

# ---------------------------------------------------------------------------
# NumPy include directory (adapted from SciPy's detection logic).
# For cross-compilation the path can be provided through the Meson cross file:
#   [properties]
#   numpy-include-dir = /path/to/numpy/_core/include
# Otherwise, query the active Python environment.
# ---------------------------------------------------------------------------
incdir_numpy = meson.get_external_property('numpy-include-dir', 'not-given')
if incdir_numpy == 'not-given'
  incdir_numpy = run_command(
    py,
    [
      '-c',
      '''import os
import numpy as np
try:
  incdir = os.path.relpath(np.get_include())
except Exception:
  incdir = np.get_include()
print(incdir)
'''
    ],
    check: true
  ).stdout().strip()
endif

inc_np = include_directories(incdir_numpy, is_system: true)

# Disable deprecated NumPy C API features for forward compatibility.
# This define only concerns C code, so it is passed as c_args to the
# extension modules instead of being attached to the dependency.
numpy_nodepr_api = ['-DNPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION']
np_dep = declare_dependency(include_directories: inc_np)

_numpy_dep = dependency('numpy')

# ---------------------------------------------------------------------------
# f2py: include directory and fortranobject.c (f2py's C runtime, which must
# be compiled together with every f2py extension module).
# ---------------------------------------------------------------------------
incdir_f2py = run_command(
  py,
  [
    '-c',
    '''import os
import numpy.f2py
try:
  incdir = os.path.relpath(numpy.f2py.get_include())
except Exception:
  incdir = numpy.f2py.get_include()
print(incdir)
'''
  ],
  check: true
).stdout().strip()

inc_f2py = include_directories(incdir_f2py, is_system: true)
fortranobject_c = incdir_f2py / 'fortranobject.c'

# libquadmath is needed by some gfortran setups (see numpy gh-25000).
quadmath_dep = fc.find_library('quadmath', required: false)

# ---------------------------------------------------------------------------
# Fortran flags.
# Warning: -march=native produces non-portable binaries (e.g. wheels).
# ---------------------------------------------------------------------------
add_project_arguments(
  fc.get_supported_arguments('-march=native'),
  language: 'fortran'
)

# ---------------------------------------------------------------------------
# Build information
# ---------------------------------------------------------------------------
compilers = {
  'C': cc,
  'FORTRAN': fc,
}

machines = {
  'HOST': host_machine,
  'BUILD': build_machine,
}

conf_data = configuration_data()

conf_data.set('NUMPY_NAME', _numpy_dep.name())
conf_data.set('NUMPY_VERSION', _numpy_dep.version())
conf_data.set('NUMPY_TYPE_NAME', _numpy_dep.type_name())

# Compiler information
foreach name, compiler : compilers
  conf_data.set(name + '_COMP', compiler.get_id())
  conf_data.set(name + '_COMP_LINKER_ID', compiler.get_linker_id())
  conf_data.set(name + '_COMP_VERSION', compiler.version())
  conf_data.set(name + '_COMP_CMD_ARRAY', ', '.join(compiler.cmd_array()))
  conf_data.set(name + '_COMP_ARGS', ', '.join(
      get_option(name.to_lower() + '_args')
    )
  )
  conf_data.set(name + '_COMP_LINK_ARGS', ', '.join(
      get_option(name.to_lower() + '_link_args')
    )
  )
endforeach

# Machine CPU and system information
foreach name, machine : machines
  conf_data.set(name + '_CPU', machine.cpu())
  conf_data.set(name + '_CPU_FAMILY', machine.cpu_family())
  conf_data.set(name + '_CPU_ENDIAN', machine.endian())
  conf_data.set(name + '_CPU_SYSTEM', machine.system())
endforeach

conf_data.set('CROSS_COMPILED', meson.is_cross_build().to_string())

# Python information
conf_data.set('PYTHON_PATH', py.full_path().replace('\\', '/'))
conf_data.set('PYTHON_VERSION', py.language_version())

# All Python sources and extension modules live under src/tolomeo.
subdir('src/tolomeo')
