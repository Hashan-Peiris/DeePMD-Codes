#!/bin/bash
#SBATCH -J DP_2-test 
#SBATCH -p gpucompute-a40  #SmeuResearch,defq,Standard,normal
##SBATCH -w chpc[097] 		#or --nodelist=node0xx Assigns the said node 	##40 for 143/144 162/162 166/167 ##32 for 164/165 and SmeuC
##SBATCH --constraint="[Scalable]"
#SBATCH --exclude=chpc[011-012,014-017]
##SBATCH  --exclusive		
#SBATCH -t 50:00:00
#SBATCH --gres=gpu:1
#SBATCH  -N 1        
#SBATCH  -n 1        
##SBATCH --mincpus=16        #min cpus per node
##SBATCH --use-min-nodes      
#SBATCH --mem-per-cpu=64GB 
##SBATCH --contiguous
##SBATCH --dependency=afterany:4802239
##SBATCH --account=sbi121

##SBATCH  --mail-user=mpeiris1@binghamton.edu
##SBATCH --mail-type=begin  # email me when the job starts
##SBATCH  --mail-type=end     # email me when the job finishes

#omp_threads=4 
export OMP_NUM_THREADS=1 
mypath="$(pwd)"

# path=$mypath                       # path without last part
# basename='target'                       # last part

# targets=( $path/${basename}* )          # all dirs in an array
# lastdir=${targets[@]: (-1):1}           # select path from last entry
# echo $lastdir
# lastdir=${lastdir##*/}                  # select filename
# lastnumber=${lastdir/$basename/}        # remove 'target'
# lastnumber=00$(( 10#$lastnumber + 1 ))  # increment number (base 10), add leading zeros

# mkdir $path/$basename${lastnumber: -3}  # make dir; last 3 chars from lastnumber

# cp $path/$lastdir/CONTCAR 		$path/$basename${lastnumber: -3}/POSCAR
# cp $path/$lastdir/KPOINTS		$path/$basename${lastnumber: -3}/KPOINTS
# cp $path/$lastdir/POTCAR		$path/$basename${lastnumber: -3}/POTCAR
# cp $path/$lastdir/INCAR			$path/$basename${lastnumber: -3}/INCAR
# cp $path/$lastdir/CHGCAR		$path/$basename${lastnumber: -3}/CHGCAR 
# cp $path/$lastdir/ICONST		$path/$basename${lastnumber: -3}/ICONST
# cp $path/$lastdir/ML_ABN		$path/$basename${lastnumber: -3}/ML_AB 
# cp $path/$lastdir/ML_FFN		$path/$basename${lastnumber: -3}/ML_FF  
# cp $path/$lastdir/job_control.txt       $path/$basename${lastnumber: -3}/job_control.txt
# cp $path/$lastdir/input.json			$path/$basename${lastnumber: -3}/input.json
# cp $path/$lastdir/*						$path/$basename${lastnumber: -3}/
# #cp $path/$lastdir/input.json		$path/$basename${lastnumber: -3}/input.json


# cd $path/$basename${lastnumber: -3}

#############################################################################################
echo "--------------------------------------------------------------------" | tee -a /home/mpeiris1/JOB.txt $mypath/JOB.txt
START_TIME=$(date +%s)
echo "Starting on $(date)" | tee -a /home/mpeiris1/JOB.txt $mypath/JOB.txt
echo "Starting: $SLURM_JOB_ID, $SLURM_JOB_NAME, $SLURM_SUBMIT_DIR" | tee -a /home/mpeiris1/JOB.txt $mypath/JOB.txt
echo "STATing: SPART = $SLURM_JOB_PARTITION, SNODES = $SLURM_JOB_NODELIST", SNODES = $SLURM_JOB_NUM_NODES, SNT = $SLURM_NTASKS, CPU-NODE = $SLURM_CPUS_ON_NODE, T-N = $SLURM_TASKS_PER_NODE | tee -a /home/mpeiris1/JOB.txt $mypath/JOB.txt
echo "--------------------------------------------------------------------" | tee -a /home/mpeiris1/JOB.txt $mypath/JOB.txt
echo "	" | tee -a /home/mpeiris1/JOB.txt $mypath/JOB.txt
#############################################################################################
sleep 1  

#module reset
#ml miniforge
conda init 
conda activate dp-cpu

# export OMPI_MCA_btl_vader_single_copy_mechanism=none
# export TMPDIR=/scratch/$USER/job_$SLURM_JOBID/  # Ensure this is local or has enough space
# export OMPI_MCA_btl="^vader"
# export OMPI_MCA_pml=ucx
# export OMPI_MCA_btl_sm_size=268435456
 
#NEW_VAR=$(( 8 * $SLURM_NTASKS ))
#echo $NEW_VAR  # This will output 16

export OMP_NUM_THREADS=1
export TF_INTRA_OP_PARALLELISM_THREADS=1 #$NEW_VAR 
export TF_INTER_OP_PARALLELISM_THREADS=1 
export DP_AUTO_PARALLELIZATION=1
export DP_INFER_BATCH_SIZE=8192

# # Disable MPI threads in Horovod
# export HOROVOD_MPI_THREADS_DISABLE=1

# # Set CUDA Toolkit root directory
# export CUDAToolkit_ROOT=$HOME/SOFTWARE/DeepMD/CUDA
# export PATH=$CUDAToolkit_ROOT/bin:$PATH
# export LD_LIBRARY_PATH=$CUDAToolkit_ROOT/lib64:$LD_LIBRARY_PATH

# # Set Horovod NCCL home directory
# export HOROVOD_NCCL_HOME=/data/home/mpeiris1/SOFTWARE/DeepMD/CNNL/nccl_2.23.4-1+cuda12.6_x86_64

# # Configure Horovod to use NCCL and TensorFlow
# export HOROVOD_GPU_ALLREDUCE=NCCL
# export HOROVOD_WITH_TENSORFLOW=1

#python runDP.py
#dp train input.json

mkdir 1.FC 
mv -v ./* ./1.FC/
mkdir 2.TEST_DATA

echo "Copying data from Version 1 of TEST DATA...!!"
#cp -rv /data/home/mpeiris1/VASP/BATTERY_STUFF/Ca-S_Projects/1.Ca-S_Reaction/8.DeepMD/1.Test/7.TEST_MODELS/0.TEST_DATA/1.Ver_1/deepmd_data ./2.TEST_DATA/
#cp -v  /data/home/mpeiris1/VASP/BATTERY_STUFF/Ca-S_Projects/1.Ca-S_Reaction/8.DeepMD/1.Test/7.TEST_MODELS/0.TEST_DATA/1.Ver_1/

ln -s /data/home/mpeiris1/VASP/BATTERY_STUFF/Ca-S_Projects/1.Ca-S_Reaction/8.DeepMD/1.Test/7.TEST_MODELS/0.TEST_DATA/1.Ver_1/deepmd_data ./2.TEST_DATA/
ln -s /data/home/mpeiris1/VASP/BATTERY_STUFF/Ca-S_Projects/1.Ca-S_Reaction/8.DeepMD/1.Test/7.TEST_MODELS/0.TEST_DATA/1.Ver_1/scripts/* ./2.TEST_DATA/

cd ./1.FC/
dp freeze -o graph.pb ; dp compress -i graph.pb -o graph-compress.pb
cp ./{graph.pb,graph-compress.pb} ../2.TEST_DATA/

cd ../2.TEST_DATA
dp test -m graph.pb -s ./deepmd_data/ -d RESULTS

conda activate SciFy2 
python Plot.py
python Outliers.py 

#############################################################################################
echo "--------------------------------------------------------------------" | tee -a /home/mpeiris1/JOB.txt $mypath/JOB.txt
echo "Finishing on $(date)" | tee -a /home/mpeiris1/JOB.txt $mypath/JOB.txt
echo "Finishing: $SLURM_JOB_ID, $SLURM_JOB_NAME, $SLURM_SUBMIT_DIR" | tee -a /home/mpeiris1/JOB.txt $mypath/JOB.txt
echo "STATing: SPART = $SLURM_JOB_PARTITION, SNODES = $SLURM_JOB_NODELIST", SNODES = $SLURM_JOB_NUM_NODES, SNT = $SLURM_NTASKS, CPU-NODE = $SLURM_CPUS_ON_NODE, T-N = $SLURM_TASKS_PER_NODE | tee -a /home/mpeiris1/JOB.txt $mypath/JOB.txt
echo "Total Time Elapsed: $(date -ud "@$(($(date +%s) - $START_TIME))" +%T) (HH:MM:SS)" |tee -a /home/mpeiris1/JOB.txt $mypath/JOB.txt
echo "--------------------------------------------------------------------" | tee -a /home/mpeiris1/JOB.txt $mypath/JOB.txt
echo "	" | tee -a /home/mpeiris1/JOB.txt $mypath/JOB.txt
#############################################################################################
