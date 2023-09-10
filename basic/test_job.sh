#!/bin/bash
#Set job requirements
#SBATCH --nodes=1 --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=thin
#SBATCH --time=10:00:00

#mkdir /scratch-shared/$USER

# Create output directory on scratch
#mkdir /scratch-shared/$USER/output

mkdir "$TMPDIR"/output_dir

chmod +x  job_None_None.sh
chmod u+x $HOME/abm/basic/run.py
#Execute program located in $HOME

module load miniconda/3
source activate MA
python $HOME/abm/basic/run.py --clf star_is1.pkl --expi appeal --out "$TMPDIR"/output_dir  --star is1

cp -r "$TMPDIR"/output_dir $HOME
