
# Shell script to install all the needed things 

export CUDA_LAUNCH_BLOCKING=1

pip install -r requirements.txt

pip install --upgrade torch torchvision torchaudio

python main.py

