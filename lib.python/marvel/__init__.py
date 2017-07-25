
import os
import marvel.config

if not os.path.exists(marvel.config.PATH_BASE):
    print("ERROR: marvel.conf.PATH_BASE is set to {} which doesn't exist".format(marvel.config.PATH_BASE))
else:
    if not os.path.exists(marvel.config.PATH_BIN):
        print("ERROR: marvel.conf.PATH_BIN is set to {} which doesn't exist".format(marvel.config.PATH_BIN))

    if not os.path.exists(marvel.config.PATH_SCRIPTS):
        print("ERROR: marvel.conf.PATH_SCRIPTS is set to {} which doesn't exist".format(marvel.config.PATH_SCRIPTS))
