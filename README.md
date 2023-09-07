# rlx: domain general, RL-driven graph transformation

Python version 3.9.10

1. build rust-lib via - https://pyo3.rs/v0.16.4/
    mostly just do `maturin develop` inside the rust-lib folder
    (if "error: linker `cc` not found", then 'sudo apt install build-essential')

2. build rlx (as a python package)
    install requirements.txt and `pip install -e rlx/`

