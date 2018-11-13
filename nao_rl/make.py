"""
Contains main functions for launching applications (VREP, NaoQI)
and 'make' function for returning an environment
"""


import subprocess, time, os
import nao_rl.settings as s

def destroy_instances():
    print("Destroying all previous VREP and NaoQI instances...")
    subprocess.Popen('pkill vrep'.split())
    subprocess.Popen('pkill naoqi-bin'.split())
    time.sleep(3)


def start_vrep(sim_port, path, exit_after_sim=False, headless=True):
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
    command += ' -s'                     # Start sim 
    command += ' ' + path                # Scene name
    command += ' &'                      # Non-blocking call
   
    # Call the process and start VREP
    DEVNULL = open(os.devnull, 'wb')
    subprocess.Popen(command.split(), stdout=DEVNULL, stderr=DEVNULL)


def start_naoqi(ports):
    """
    Launches a new virtual NAO at a specified port
    """
    if type(ports) is not list:
        ports = [ports]
    command = ''
    for instance in ports:
        command += s.CHORE_DIR + '/naoqi-bin' + \
                  ' -p {}'.format(instance) + ' & '

    print '==========================================================================='
    print command
    subprocess.Popen(command.split())
    time.sleep(5)

def make(env_name, sim_port, nao_port=None, headless=True, reinit=False):
    """
    Launches VREP, Naoqi at specified ports
        arguments:
            [reinit]   : if true sends 'pkill vrep' and 'pkill naoqi-bin'
            [headless] : start VREP with a GUI
    """
    if reinit:
        destroy_instances()

    ###########################
    ### CUSTOM ENVIRONMENTS ###
    ###########################

    if env_name == 'nao_bipedal':
        from nao_rl.environments import NaoWalking
        path = s.SCENES + '/nao_test.ttt'

        print "Launching V-REP at port {}".format(sim_port)
        start_vrep(sim_port, path, headless=headless)
        if headless: time.sleep(1.5)
        else: time.sleep(5)
        env = NaoWalking(s.LOCAL_IP, sim_port, nao_port, path)

    if env_name == 'nao-bipedal2':
        from nao_rl.environments import NaoWalking2
        path = s.SCENES + '/nao_test2.ttt'
        print "Launching V-REP at port {}".format(sim_port)
        start_vrep(sim_port, path, headless=headless)
        if headless: time.sleep(1.5)
        else: time.sleep(5)
        env = NaoWalking2(s.LOCAL_IP, sim_port, nao_port, path)


    elif env_name == 'nao_tracking':
        from nao_rl.environments import NaoTracking
        path = s.SCENES + '/nao_ball.ttt'
        print "Launching V-REP at port {}".format(sim_port)
        start_vrep(sim_port, path, headless=headless)
        time.sleep(2)
        env = NaoTracking(s.LOCAL_IP, sim_port, nao_port, path)

    else:
        raise RuntimeError('No such environment.')

    return env


