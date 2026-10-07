# Installing pyconvino on macOS with an x86_64 Python

PyPI has no JAX ≥ 0.10 for x86_64 macOS (the last x86_64 macOS wheel is
0.4.38), so there `pip install pyconvino` cannot find a matching `jax`. This
also hits Apple-silicon Macs that run an x86_64 Python under Rosetta (e.g. an
old Intel Homebrew or Mambaforge install). Check your Python:

```bash
python -c "import platform; print(platform.machine())"   # arm64 or x86_64
```

On Apple silicon, if it prints `x86_64`:

- **Use a native arm64 Python (recommended).** For example a native
  Miniforge, or a conda env forced to arm64:
  ```bash
  CONDA_SUBDIR=osx-arm64 conda create -n pyconvino python=3.12 pip
  conda activate pyconvino
  conda config --env --set subdir osx-arm64
  pip install pyconvino
  ```

On an Intel Mac, or if you must keep the x86_64 Python:

- **Take JAX from conda-forge**, which still builds it for x86_64 macOS (it
  also runs under Rosetta, but slower than native):
  ```bash
  conda install -c conda-forge "jax>=0.10" numpy scipy
  pip install pyconvino
  ```

Linux (x86_64 and aarch64) and native arm64 macOS work with plain
`pip install pyconvino`.
