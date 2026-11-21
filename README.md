# Tolomeo

`tolomeo` is derived from [ROMA](https://github.com/ilcarogna/MapMaker-ROMA) by Giancarlo de Gasperis ([de Gasperis et al., 2005](https://doi.org/10.1051/0004-6361:20042512)), released under GPL-3.0-or-later; the derived parts are distributed here under MPL-2.0 with the author's permission.

## Install Guide

### From GitHub

`tolomeo` manages its dependencies using `uv`. If you haven't installed it yet, you can follow the installation guide provided by [Astral](https://docs.astral.sh/uv/getting-started/installation/).

Once `uv` is ready, clone this repository locally and navigate into the project directory:

```bash
git clone https://github.com/giusesorrentino/tolomeo.git
cd tolomeo
```

Then, create a virtual environment:

```bash
uv venv
```

Now that the virtual environment has been created, activate it using the following command on macOS/Linux:

```bash
source .venv/bin/activate
```

or, on Windows:

```powershell
.venv\Scripts\activate
```

If everything went smoothly, your terminal should look like this:

```bash
(tolomeo) $
```

> [!NOTE]
> Remember to reactivate the environment every time you open a new terminal session!

With the environment active, install all the core and development dependencies:

```bash
uv sync
```
