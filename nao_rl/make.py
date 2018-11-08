"""
Contains main functions for launching applications (VREP, NaoQI)
and 'make' function for returning an environment
"""


import subprocess, time
import nao_rl.settings as s

def start_vrep(sim_port, path, exit_after_sim=True, headless=True):
    """
    Launches a new V-REP instance using subprocess
    If you want to connect to an existing instance, use self.connect() with a correct port instead
        arguments:
            [-h] : runs in headless mode (without GUI)
    """

    command = 'bash ' + s.VREP_DIR + '/vrep.sh'  + \
            ' -gREMOTEAPISERVERSERVICE_{}_TRUE_TRUE'.format(sim_port) # Start remote API at a specified port

    # Additional arguments
    if headless:       command += ' -h'
    if exit_after_sim: command += ' -q'  
    command += ' ' + path                # Scene name
   

    # Call the process and start VREP
    subprocess.Popen(command.split())


def start_naoqi(port):
    """
    Launches a new virtual NAO at a specified port
    """
    command = s.CHORE_DIR + '/naoqi-bin' + \
              ' -p {}'.format(port)

    subprocess.Popen(command.split())


def make(env_name, sim_port, nao_port, headless=True, reinit=False):
    """
    Launches VREP, Naoqi at specified ports
        arguments:
            [reinit]   : if true sends 'pkill vrep' and 'pkill naoqi-bin'
            [headless] : start VREP with a GUI
    """
    if reinit:
        print("Destroying all previous VREP and NaoQI instances...")
        subprocess.Popen('pkill vrep'.split())
        subprocess.Popen('pkill naoqi-bin'.split())
        time.sleep(3)

    print "Launching NaoQI at port {}...".format(nao_port)
    start_naoqi(nao_port)
    time.sleep(1)

    if env_name == 'nao_bipedal':
        from nao_rl.environments import NaoWalking
        path = s.SCENES + '/nao_test.ttt'

        print "Launching V-REP at port {}".format(sim_port)
        start_vrep(sim_port, path, headless=headless)
        time.sleep(1)
        env = NaoWalking(s.LOCAL_IP, sim_port, nao_port, path)

    elif env_name == 'nao_tracking':
        from nao_rl.environments import NaoTracking
        path = s.SCENES + '/nao_ball.ttt'
        print "Launching V-REP at port {}".format(sim_port)
        start_vrep(sim_port, path, headless=headless)
        time.sleep(1)
        env = NaoTracking(s.LOCAL_IP, sim_port, nao_port, path)

    else:
        raise RuntimeError('No such environment.')

    return env


