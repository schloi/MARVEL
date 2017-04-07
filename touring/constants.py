
from marvel import LAS

### flags
#

E_BACKBONE  = (LAS.OVL_CUSTOM << 0)
E_INVERSION = (LAS.OVL_CUSTOM << 1)

V_BACKBONE  = (1 << 0)
V_DISCARD   = (1 << 1)
V_DEAD_END  = (1 << 2)
V_VISITED   = (1 << 3)
V_PATH_END  = (1 << 4)
V_RETRY     = (1 << 5)
V_MODULE    = (1 << 6)

#
###

### settings
#

MAX_BB_DISTANCE                   = 3

DEF_PATH_LOOKAHEAD                = 6
PATH_LOOKAHEAD_INCREASE_INVERSION = 3

#
###


