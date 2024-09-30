#!/bin/bash
#SBATCH --job-name=gpt2-large-sft
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4

# Print a message to indicate the start of the process
echo "Activating the virtual environment and starting the training process..."

# Set the environment variable to use GPU 0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4

# Activate the Python virtual environment
source env/bin/activate

# added configuration for FSDP
ulimit -n 64000

# run python scripts with given parameters
python -u train.py\
        model=gpt2-large\
        datasets=[imdb]\
        loss=dpo\
        loss.beta=0.1\
        exp_name=testing_length\
        gradient_accumulation_steps=2\
        batch_size=64\
        eval_batch_size=32\
        trainer=FSDPTrainer\
        sample_during_eval=false\
        eval_every=5000\
        model.fsdp_policy_mp=bfloat16\
        model.archive=.cache/vck/debug_only_2024-09-16_20-08-56_240661/LATEST/policy.pt\


# Print a message to indicate the completion of the training
echo "Training process completed."
