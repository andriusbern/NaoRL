"""
Author: Andrius Bernatavicius, 2018

Contains main functions for launching applications (VREP, NaoQI)
and 'make' function for creating and opening the custom environments
"""


import subprocess, time, os
import importlib
from nao_rl import settings


def destroy_instances():
    """
    Destroys all the current instances of Vrep and Naoqi-bin
    """
    print "Destroying all previous VREP and NaoQI instances..."
    subprocess.Popen('pkill vrep'.split())
    subprocess.Popen('pkill naoqi-bin'.split())
    time.sleep(1)


def start_vrep(sim_port, path, exit_after_sim=False, headless=True, verbose=False):
    """
    Launches a new V-REP instance using subprocess
    If you want to connect to an existing instance, use self.connect() with a correct port instead
        arguments:
            [sim_port]      : VREP simulation port, set by '-gREMOTEAPISERVERSERVICE_port_TRUE_TRUE'
            [path]          : path of the V-REP scene (.ttt format) to run
            [exit_ater_sim] : corresponds to command line argument '-q' (exits after simulation)
            [headless]      : corresponds to '-h' (runs without a GUI)
            [verbose]       : suppress prompt messages
    """

    command = 'bash ' + settings.VREP_DIR + '/vrep.sh'  + \
            ' -gREMOTEAPISERVERSERVICE_{}_TRUE_TRUE'.format(sim_port) # Start remote API at a specified port

    # Additional arguments
    if headless:
        command += ' -h'
    if exit_after_sim:
        command += ' -q'

    command += ' -s'                     # Start sim
    command += ' ' + path                # Scene name
    command += ' &'                      # Non-blocking call

    # Call the process and start VREP
    print "Launching V-REP at port {}".format(sim_port)
    DEVNULL = open(os.devnull, 'wb')
    if verbose:
        subprocess.Popen(command.split())
    else:
        subprocess.Popen(command.split(), stdout=DEVNULL, stderr=DEVNULL)


def start_naoqi(ports):
    """
    Launches a new virtual NAO provided by 'Choregraphe' software suite at a specified port 
    """
    if type(ports) is not list:
        ports = [ports]
    command = ''
    for instance in ports:
        command += settings.CHORE_DIR + '/naoqi-bin' + \
                  ' -p {}'.format(instance) + ' & '

    print '==========================================================================='
    print command
    subprocess.Popen(command.split())
    time.sleep(5)


def make(env_name, sim_port=None, nao_port=None, headless=True, reinit=False, **kwargs):
    """
    Launches VREP, Naoqi at specified ports
        arguments:
            [reinit]   : if true sends 'pkill vrep' and 'pkill naoqi-bin'
            [headless] : start VREP with a GUI
    """
    if reinit:
        destroy_instances()
    
    # Try to get the environment object
    try:
        module = importlib.import_module('nao_rl.environments' + '.' + env_name)
        env_object = getattr(module, env_name)
    except:
        print "No such environment!"

    if sim_port is None:
        sim_port = settings.SIM_PORT

    env = env_object(settings.LOCAL_IP, sim_port, kwargs)
    start_vrep(sim_port, env.path, headless=headless)

    if headless: time.sleep(1.5)
    else:        time.sleep(5)

    env.initialize() # Connect python client to the new V-REP instance
    settings.SIM_PORT -= 1

    return env


def save_model(filename, object, experiment_name):
    """
    Saves a model in the trained_models dir
    """

    import pickle
    loc = s.MAIN_DIR + '/trained_models'
    suffix = '/model_{}'.format(experiment_name)
    file = loc + suffix + '.pickle'
    with open(file, 'wb') as f:
        pickle.dump(object, f)


def load_model(filename):
    """
    Loads a trained model from a pickle file
    """
    import pickle

    with open(filename, 'wb') as f:
        return pickle.load(f)




