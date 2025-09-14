
# Shell script to install all the needed things 

export CUDA_LAUNCH_BLOCKING=1

python -m pip install --upgrade pip

pip install -r requirements.txt

pip install --upgrade torch torchvision torchaudio

#python testGPU.py

python main.py

