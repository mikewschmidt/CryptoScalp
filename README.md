# CryptoScalp

Trying to train a deep learning model to predict prices in a short time frame

# Instructions for Running the Project

## Step 0: Clone this repository

```bash
git clone https://github.com/mikewschmidt/CryptoScalp.git
cd CryptoScalp
sudo chmod +x run.sh
./run.sh  ## This will install requirements and update things and run 3 month of data
```

## Step 1: Install Dependencies

Before running the main script, you need to install the required packages listed in `requirements.txt`. Open your terminal and navigate to the project directory, then run:

```bash
pip install -r requirements.txt
```

## Step 2: Test GPU Availability

To ensure that your GPU is functioning correctly, run the testGPU.py script. This script will check if PyTorch can access the GPU. Execute the following command in your terminal:

```bash
python testGPU.py
```

### Expected Output

If the GPU is available, you should see a message indicating that the GPU is being used. If not, the script will inform you that it will use the CPU.

## Step 3: Run the Main Script

Once you have confirmed that the GPU is working, you can run your main script. Use the following command:

```bash
python main.py
```
