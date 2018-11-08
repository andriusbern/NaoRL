import os

# Directories

MAIN_DIR = os.path.dirname(os.path.realpath(__file__))
SCENES = MAIN_DIR + '/scenes'
TRAINED_MODELS = MAIN_DIR + '/trained_models'

# Addresses and ports
NAO_PORT = 5995
SIM_PORT = 19998
LOCAL_IP = '127.0.0.1'

RNAO_IP = 9559
RNAO_PORT = '192.168.1.175'
VREP_DIR = '/home/andrius/thesis/software/V-REP_PRO_EDU_V3_4_0_Linux'
CHORE_DIR = '/home/andrius/thesis/software/choregraphe/bin'