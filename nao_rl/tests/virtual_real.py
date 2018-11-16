# Code snippets to test real and virtual NAO

## Deprecated

from nao_rl.utils import VirtualNAO, RealNAO
import nao_rl.settings as s
from nao_rl.utils import image_processing as im

import cv2 as cv
import numpy as np
import time

real = RealNAO(s.RNAO_IP, s.RNAO_PORT)
virtual = VirtualNAO(s.LOCAL_IP, s.NAO_PORT)

