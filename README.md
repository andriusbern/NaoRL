# Summary
A package that contains several reinforcement learning environments featuring NAO in VREP simulation software.

## Requirements
### Software
1. Python 2.7 and python-virtualenv
2. [VREP v3.4.0](http://coppeliarobotics.com/files/V-REP_PRO_EDU_V3_4_0_Linux.tar.gz) - Robot simulation software by Copellia Robotics
3. [Choregraphe Suite v2.1.2](https://community.ald.softbankrobotics.com/en/dl/ZmllbGRfY29sbGVjdGlvbl9pdGVtLTQyNS1maWVsZF9zb2Z0X2RsX2V4dGVybmFsX2xpbmstMC01MzY2MjU%3D?width=500&height=auto) - for creating a virtual NAO (**requires registering**) (By default installed in */opt/Aldebaran/Choregraphe Suite 2.1*)



1. [Python NAOqi SDK v2.1.2](https://community.ald.softbankrobotics.com/en/dl/ZmllbGRfY29sbGVjdGlvbl9pdGVtLTQ1My1maWVsZF9zb2Z0X2RsX2V4dGVybmFsX2xpbmstMC1lZWFmOGU%3D?width=500&height=auto) - Libraries provided by Softbank robotics for NAO control (**requires registering**)

### Python packages
numpy, tensorflow, keras, opencv-python

# Installation
*(Tested on Ubuntu 18.04)*

**1. Clone the repository**
   
```
git clone https://github.com/andriusbern/nao_rl
cd nao-rl
```
**2. Create and activate the virtual environment**
   
```
virtualenv env
source env/bin/activate
```

**3. Install the package and the required libraries**
```
python setup.py install
```
*(Might need to run twice)*
This script will also prompt you to enter the paths to:
1. VREP
2. Choregraphe
3. PyNAOQI SDK

**4. Launch the virtual NAO and VREP.**
```
chmod u+x vrep.sh
./vrep.sh
```
By default this script launches:
- VREP
- naoqi-bin at port 5995 - to simulate a virtual nao that can be controlled with PYNAOQI Proxies

# Examples

You can try one of the example scripts by running:

**Learning a bipedal gait**
```
python nao_rl/learning/ddpg_nao_bipedal_gait.py
```


**Learning to track a ball**
```
python nao_rl/learning/ddpg_nao_tracking.py
```
Both scripts use Deep Deterministic Policy Gradient (DDPG) algorithm.
Currently the scripts are not optimized and fail to produce good policies.