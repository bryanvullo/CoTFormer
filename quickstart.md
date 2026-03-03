# Quickstart

## Iridis X

log into iridis X:

`ssh <username>@loginx001.iridis.soton.ac.uk`

Enter password... Logged in!

## Setting up Iridis (Run once)

Clone Repo: `git clone https://github.com/bryanvullo/CoTFormer`

Move into Folder: `cd /scratch/<username>`

> Note: run `which python` to see you're on Iridis's standard env.

Make venv: `conda create -n 'cotformer-env'`

Activate venv: `conda activate cotformer-env`

> Note: run `which python` to see you're in new venv, ready to install packages.

> Note: run `deactivate` to deactivate the venv

Install dependencies: `pip install -r /home/<username>/CoTFormer/requirements.txt`

## Run job scripts

Customise job.sh as you'd like and run:

`cd /home/<username>/CotFormer`

`sbatch job.sh`