# Tolomeo

`tolomeo` is derived from [ROMA](https://github.com/ilcarogna/MapMaker-ROMA) ([de Gasperis et al., 2005](https://doi.org/10.1051/0004-6361:20042512)), released under GPL-3.0-or-later; the derived parts are distributed here under MPL-2.0 with the author's permission.

## Installation

### With pip

To use `tolomeo` without changing its code, install it directly from GitHub, with Python 3.12 or later:

```bash
pip install git+https://github.com/giusesorrentino/tolomeo.git
```

### For development

`tolomeo` manages its dependencies with [`uv`](https://docs.astral.sh/uv/). If you do not have it yet, follow the [installation guide](https://docs.astral.sh/uv/getting-started/installation/) by Astral.

Clone the repository and enter the project directory:

```bash
git clone https://github.com/giusesorrentino/tolomeo.git
cd tolomeo
```

Create a virtual environment:

```bash
uv venv
```

and activate it on macOS and Linux:

```bash
source .venv/bin/activate
```

or on Windows:

```powershell
.venv\Scripts\activate
```

The prompt of the terminal now starts with the name of the environment:

```bash
(tolomeo) $
```

> [!NOTE]
> Activate the environment again in every new terminal session.

With the environment active, install `tolomeo` with its core and development dependencies:

```bash
uv sync
```
