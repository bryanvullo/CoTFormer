# Quickstart

## Iridis X

log into iridis X:

`ssh <username>@loginx001.iridis.soton.ac.uk`

Enter password... Logged in!

## Setting up Iridis (Run once)

Clone Repo: `git clone https://github.com/bryanvullo/CoTFormer`

## Enviromnent

Move into Folder: `cd /scratch/<username>`

Install Conda:
`wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh`
`bash Miniforge3-Linux-x86_64.sh -b -p ./mambaforge`
`echo 'source ./mambaforge/etc/profile.d/conda.sh' >> ./.bashrc`
`source ./.bashrc`

> Note: run `which python` to see you're on Iridis's standard env.

Make env: `conda create -n 'cotformer-env'`

Activate env: `conda activate cotformer-env`

> Note: run `which python` to see you're in new venv, ready to install packages.

> Note: run `deactivate` to deactivate the venv

Install dependencies: `pip install -r /home/<username>/CoTFormer/requirements.txt`

## Run job scripts

Customise job.sh as you'd like and run:

`cd /home/<username>/CotFormer`

`sbatch job.sh`