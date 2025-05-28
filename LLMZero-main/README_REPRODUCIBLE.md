## Reproduce result

#### Prerequisite 

- Linux with Nvidia GPU



#### Setup and run

1. Setup conda environment

   ```bash
   conda create -n llmzero_reproducible python=3.11
   conda activate llmzero_reproducible
   ```

2. Install dependencies

   ```bash
   pip install -r requirements_reproducible.txt
   ```

3. Install `pyRDDLGym`

   ```bash
   pip install -q git+https://github.com/tasbolat1/pyRDDLGym.git --force-reinstall
   pip install numpy==1.24.2 --force-reinstall
   ```

4. Run the script

   ```sh
   sh reproducible.sh
   ```