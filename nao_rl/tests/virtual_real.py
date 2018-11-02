# Code snippets to test real and virtual NAO

from nao_rl.utils import VirtualNAO, RealNAO
import nao_rl.settings as s

real = RealNAO(s.RNAO_IP, s.RNAO_PORT)
virtual = VirtualNAO(s.LOCAL_IP, s.NAO_PORT)

