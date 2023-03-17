#BSUB -W 3:30
#BSUB -n 2
#BSUB -R "span[ptile=2]"
#BSUB -M 10GB
#BSUB -J sample
#BSUB -e /users/kin8hb/%J.err
#BSUB -o /users/kin8hb/%J.err

module load anaconda3
source activate tensorflow-2

jupyter nbconvert --to python "/users/kin8hb/sample_notebook.ipynb"
python "/users/kin8hb/sample_notebook.py"
