# Setting up

## Set up Locally

- Create a new Conda environment, and activate.
```bash
conda env create -f env.yml
conda activate aml-env

```
- Install Pytorch and Pytorch Geometric
```bash
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pyg -c pyg
pip install -r requirements.txt
```

## Set up for DAIC

Use the `Apptainer.def` file to change the data stored and behavior of the container. Currently, it is set up to run the same commands as the ones indicated above. 

To create the image file, run 
```bash
apptainer build <container_name>.sif Apptainer.def
```

To deploy on DAIC, you can use the `sbatch.execute.slurm` file configured for the DAIC cluster. Don't forget to change the `${container_name}` variable to the name of your container.
Then send the files (`.sif` file, `sbatch.execute.slurm`) to the DAIC cluster. You can use `scp` for that:

```bash
scp (-r) /path/to/your/project/<container_name>.sif <netid>@login.daic.tudelft.nl:/path/to/your/directory
```

After that, you can run the following command to submit the job:

```bash
sbatch sbatch.execute.slurm
```
# Data

The data needed for the experiments can be found on [Kaggle](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/data).

