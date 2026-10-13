import os
import shutil
import sys
import sysconfig
from pathlib import Path

import click
import spin

ROOT = Path(__file__).resolve().parents[1]
RUN_CMD = ["uv", "run", "--active"]


@click.command(context_settings={"ignore_unknown_options": True})
@click.argument("path", default="src/ichnoof", required=False)
@click.option("--fix", is_flag=True, default=False, help="Auto-fix fixable lint errors")
@click.option(
    "--unsafe-fixes", is_flag=True, default=False, help="Auto-fix hidden lint errors"
)
def lint(path, fix, unsafe_fixes):
    """🔦 Run ruff on the codebase.

    \b
    Examples:

    \b
    Lint the whole package:
    $ spin lint

    \b
    Lint a specific file or directory:
    $ spin lint src/ichnoof/optimize/fit.py

    \b
    Auto-fix fixable errors:
    $ spin lint --fix
    """
    target = Path(path).resolve()

    if not target.exists():
        raise click.BadParameter(f"Path '{path}' does not exist.")

    print(f"✨ Linting and formatting code in '{path}' with Ruff...")
    check_cmd = ["uvx", "ruff", "check", str(target)]

    if fix:
        check_cmd += ["--fix"]
    if unsafe_fixes:
        check_cmd += ["--unsafe-fixes"]

    format_cmd = ["uvx", "ruff", "format", str(target)]

    try:
        spin.util.run(check_cmd)
    except SystemExit:
        pass

    spin.util.run(format_cmd)


@click.command()
@click.option(
    "--install/--no-install",
    default=True,
    help="Install the package in editable mode after building.",
)
def build(install):
    """🔧 Build the package (and install it in editable mode by default).

    \b
    Examples:

    \b
    Build and install in editable mode (default):
    $ spin build

    \b
    Build only, without installing:
    $ spin build --no-install
    """

    v = sys.version_info
    suffix = "t" if sysconfig.get_config_var("Py_GIL_DISABLED") else ""
    relative_build_dir = f"build/cp{v.major}{v.minor}{suffix}"
    build_dir = ROOT / relative_build_dir

    print(
        f"🧹 Cleaning previous build directory for Python {v.major}.{v.minor}{suffix}..."
    )
    if build_dir.exists():
        shutil.rmtree(build_dir)

    if install:
        print("🔨 Building and installing in editable mode...")
        spin.util.run(
            ["uv", "pip", "install", "-e", str(ROOT), "--no-build-isolation"]
        )
    else:
        print(f"🔨 Building only in '{relative_build_dir}' (no installation)...")
        spin.util.run(RUN_CMD + ["meson", "setup", str(build_dir)])
        spin.util.run(RUN_CMD + ["meson", "compile", "-C", str(build_dir)])


@click.command()
@click.argument("paths", nargs=-1)
@click.option("-v", "--verbose", is_flag=True, default=False)
@click.option("-s", "--run-slow", is_flag=True, default=False)
def test(paths, verbose, run_slow):
    """🧪 Run the test suite with pytest.

    \b
    Examples:

    \b
    Run all tests:
    $ spin test

    \b
    Run a specific file or directory:
    $ spin test tests/unit/test_zernike.py

    \b
    Run with verbose output:
    $ spin test -v

    \b
    Run slow tests:
    $ spin test -s
    """
    print("🧪 Running tests...")

    cmd = ["uv", "run", "--active", "pytest"]
    if verbose:
        cmd += ["-v"]
    if run_slow:
        cmd += ["--run-slow"]
    if paths:
        resolved_paths = []
        for path in paths:
            p = Path(path)
            # use directly if absolute path or already relative to tests/
            if p.is_absolute() or path.startswith("tests/"):
                resolved_paths.append(path)
                continue
            # try first as a direct path relative to root
            direct = ROOT / "tests" / path
            if direct.exists():
                resolved_paths.append(str(direct))
                continue
            # fallback: recursive glob in tests/
            matches = list((ROOT / "tests").rglob(path))
            if matches:
                resolved_paths.extend(str(m) for m in matches)
            else:
                # pass as-is and let pytest throw the error
                resolved_paths.append(path)

        cmd += resolved_paths
    spin.util.run(cmd=cmd, cwd=ROOT)



@click.command()
def smoke():
    """🔥 Build and smoke-test the wheel and sdist distributions.

    Builds the package into a temporary directory, then runs the smoke
    test suite against both the wheel and the source distribution in
    an isolated environment to verify that the installed package imports
    and behaves correctly.

    \b
    Examples:

    \b
    Run the smoke test for the current Python interpreter:
    $ spin smoke
    """
    v = sys.version_info
    suffix = "t" if sysconfig.get_config_var("Py_GIL_DISABLED") else ""
    tag = f"Python {v.major}.{v.minor}{suffix}"

    whldir = ROOT / ".tolomeo" / "smoke_test" / f"cp{v.major}{v.minor}{suffix}"

    print(f"🔥 Running smoke test for {tag} (it may take a while)...")

    if whldir.exists():
        shutil.rmtree(whldir)
    whldir.mkdir(parents=True)

    print(f"📦 Building wheel and sdist for {tag}...")
    spin.util.run(
        ["uv", "build", "--out-dir", str(whldir)],
        cwd=ROOT,
    )

    wheel = next(whldir.glob("*.whl"))
    sdist = next(whldir.glob("*.tar.gz"))

    print("✅ Testing wheel installation...")
    spin.util.run(
        [
            "uv",
            "run",
            "--active",
            "--isolated",
            "--no-project",
            "--with",
            str(wheel),
            "tests/smoke_test.py",
        ],
        cwd=ROOT,
    )

    print("✅ Testing sdist installation...")
    spin.util.run(
        [
            "uv",
            "run",
            "--active",
            "--isolated",
            "--no-project",
            "--with",
            str(sdist),
            "tests/smoke_test.py",
        ],
        cwd=ROOT,
    )

    shutil.rmtree(whldir)
    print(f"🎉 Smoke test passed for {tag}!")