# fashion_ai_backend
This repo contains the python code which utilizes the ngrok for the tunneling and uses the fash_vton 1.5 for the try on.

### 1. Create Virtual environment to install the necessary packages

```bash
python -m venv .fash
# this creates a virtual environment in the name of .fash
```

### 2. Source the venv and Install the packages 
```bash
# source the venv
source .fash/scripts/activate  # for python 13
# or
source .fash/bin/activate # for python < 13


# install packages
pip install -r requirements.txt    # this installs the packages

```

### 3. uninstall and install CUDA enabled torch

```bash

pip uninstall torch torchvision

# and install cuda enabled
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130  # this install the torch with cuda enabled
```

### 4. Finally add the ngrok token in the python file and run the project

Get NGROK token from the website, and create you account and inside your authtoken you will get the token. Copy that token and paste in the ***NGROK_TOKEN = "YOUR_TOKEN_HERE" ***


and finally run the python code. The code will do the following:

1. Downloads the Fash-vton-1.5 source from the git in the name **fashion**
2. Downloads the model weights and will save in the folder **weights**
3. Loads the model with CUDA if available or with CPU (which will be slower)
4. And Last you will get a api link, use that in the app to make a connection between the app and the model.

```bash
# to run the code
python fash_backend.py
```
