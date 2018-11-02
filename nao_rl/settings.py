import os

# Directories

MAIN_DIR = os.path.dirname(os.path.realpath(__file__))
SCENES = MAIN_DIR + '/scenes'
TRAINED_MODELS = MAIN_DIR + '/trained_models'

# Addresses and ports
NAO_PORT = 5995
SIM_PORT = 19998
LOCAL_IP = '127.0.0.1'

RNAO_IP = ''
RNAO_PORT = ''