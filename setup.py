from setuptools import setup
import time, os

#################
# Configure paths

separator = "\n================================================================\n"

print separator, "Installation script for the 'nao_rl' package...", separator

if not os.path.isfile(os.getcwd() + '/settings.txt'):

    print "Directory paths for the following software need to be configured...:"
    print "VREP (version 3.4.0)\n", "Choregraphe\n", "pynaoqi-python2.7-2.1.2-17-linux library."
    paths = []

    # VREP
    print separator
    while True:
        vrep_dir = raw_input("\nPlease paste the path to your 'VREP' folder (Version 3.4.0)...\n")
        if os.path.exists(vrep_dir):
            if os.path.isfile(vrep_dir + '/vrep.sh'):
                print 'Directory added to settings.'
                paths.append(vrep_dir)
                break  
            else:
                print "'vrep.sh' is not in the provided directory..."
        else: 
            print "Invalid directory"

    # Choregraphe
    print separator
    while True:
        chore_dir = raw_input("\nPlease paste the path to your 'Choregraphe' folder...\n")
        if os.path.exists(chore_dir):
            if os.path.isfile(chore_dir + '/bin/naoqi-bin'):
                print 'Directory added to settings.'
                paths.append(chore_dir+'/bin')
                break  
            else:
                print "'naoqi-bin' is not in the provided directory..."
        else: 
            print "Invalid directory"

    # Naoqi-SDK
    print separator
    while True:
        naoqi_dir = raw_input("\nPlease paste the path to your 'Naoqi-Python SDK' folder (Version 2.1.2)...\n")
        if os.path.exists(naoqi_dir):
            if os.path.isfile(naoqi_dir + '/naoqi.py'):
                print 'Directory added to settings.'
                paths.append(naoqi_dir)
                break  
            else:
                print "'naoqi.py' is not in the provided directory..."
        else: 
            print "Invalid directory"

    # Create a settings file with directories
    with open('settings.txt', 'wb') as file:
        for line in paths:
            file.writelines(line+'\n')
    
    
    # Add naoqi to the PYTHONPATH of the virtualenv (by creating a '.pth' file in site-packages folder)
    path = os.getcwd() + '/env/lib/python2.7/site-packages/naoqi.pth'
    with open(path, 'wb') as file2:
        file2.write(paths[2])

else:
    print separator, "File 'settings.txt' already exists."


######################
# Package installation
packages = ['numpy',
            'tensorflow',
            'keras',
            'gym',
            'keras-rl',
            'opencv-python',
            'Pillow']

print separator, separator
print "Paths added successfully..."
print "Now the following required Python packages are going to be installed in the virtual environment:\n"
print packages
time.sleep(3)

setup(
    name='nao_rl',
    description='VREP RL package for NAO robot.',
    long_description='',
    version='0.1',
    packages=['nao_rl'],
    scripts=[],
    author='Andrius Bernatavicius',
    author_email='andrius.bernatavicius@gmail.com',

    url='none',
    download_url='none',
    install_requires=packages
)

print separator, "Installation complete.\n"




