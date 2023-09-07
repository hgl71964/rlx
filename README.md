# ReLgg: deep reinforcement learning for equality saturation extraction

Python version 3.9.10

1. build rust-lib via - https://pyo3.rs/v0.16.4/
    mostly just do `maturin develop` inside the rust-lib folder
    (if "error: linker `cc` not found", then 'sudo apt install build-essential')

2. build ReLgg (as a python package)
    install requirements.txt and `pip install -e python/`

# Reference:
        https://www.cl.cam.ac.uk/~ey204/pubs/MPHIL_P3/2022_Zak.pdf

