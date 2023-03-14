#BSUB -W 3:30
#BSUB -q gpu-v100
#BSUB -gpu "num=1"
#BSUB -n 2
#BSUB -R "span[ptile=2]"
#BSUB -M 150GB
#BSUB -J nextmod
#BSUB -e /users/kin8hb/%J.err
#BSUB -o /users/kin8hb/%J.err
module load anaconda3
module load cuda/10.1
module load cuda/10.2
module load cuda/11.2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64/
source activate tensorflow-2
jupyter nbconvert --to python "/users/kin8hb/Spring_2023/Programs/multifunction_ensemble/train_next_ensemble_model.ipynb"
python "/users/kin8hb/Spring_2023/Programs/multifunction_ensemble/train_next_ensemble_model.py"
