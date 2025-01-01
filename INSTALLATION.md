# Installation

Clone the repo:

```bash
git clone git@github.com:ykarmesh/stable-control-representations.git
cd stable-control-representations

git submodule update --init --recursive  # Also necessary if we updated any submodules
```

[Install Conda package manager](https://docs.conda.io/en/latest/miniconda.html). Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate stable-control-representations  # Alternatively, `direnv allow`
```

Install Diffusers library (with our custom changes):
```bash
cd third_party/diffusers
pip install -e .[torch]
```

Setup Mujoco/mj_envs/mjrl:
```bash
mkdir ~/.mujoco
# Go to https://www.roboti.us/download.html to download Mujoco library
wget https://www.roboti.us/download/mujoco200_linux.zip -P ~/.mujoco
unzip ~/.mujoco/mujoco200_linux.zip

# Go to https://www.roboti.us/license.html to obtain the key under their Free license:
wget https://www.roboti.us/file/mjkey.txt -P ~/.mujoco
```

```bash
# Install mujoco-py (GPU-compiled)
pip install -e ./third_party/mujoco-py

# Install mj_envs/mjrl
pip install -e ./third_party/mj_envs
pip install -e ./third_party/mjrl
```

Install Habitat-Lab v0.2.1 (patched to remove Tensorflow dependencies):

```bash
cd third_party/habitat-lab
python setup.py develop --all # install habitat and habitat_baselines
cd -
```

Install local packages:


```bash
pip install -e ./vc_models  # Install model-loading API
pip install -e ./benchmarks/mujoco_vc  # Install Visual IL tasks
pip install -e ./benchmarks/habitat_vc  # Install Habitat tasks
```

If you are unable to load `mujoco_py` with error `ImportError: cannot import name 'load_model_from_path' from 'mujoco_py' (unknown location)`, try running

```bash
rm -rf ~/.local/lib/python3.8/site-packages/mujoco_py
```