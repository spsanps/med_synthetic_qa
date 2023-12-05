#!/bin/bash
# File name: run_scripts_sequentially.sh

# Run the first script in the background and get its PID
python train_peft_mistral.py --inv > logs/train_peft_mistral_inv2.log 2>&1 &
PID=$!

# Wait for the first script to finish
wait $PID

# Check the exit status of the first command
if [ $? -eq 0 ]; then
  echo "First command succeeded, running the second command."
  # Run the second script in the background
  python train_peft_mistral.py > logs/train_peft_mistral2.log 2>&1 &
else
  echo "First command failed, stopping."
  exit 1
fi
