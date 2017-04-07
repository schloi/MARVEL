
from __future__ import print_function

import multiprocessing
import sys
import os
import subprocess
import time
import shlex
import glob
import types

import marvel.rawqueue
import marvel.config

class queue(marvel.rawqueue.rawqueue):

    DEFAULT_DALIGNER_THREADS = 4

    def __init__(self, db, cov, cores,
                       path_bin = marvel.config.PATH_BIN,
                       path_scripts = marvel.config.PATH_SCRIPTS,
                       checkpoint = None, finish_block_on_error = False):

        super(queue, self).__init__(cores, checkpoint = checkpoint, finish_block_on_error = finish_block_on_error)

        self.db_path = db
        self.db = os.path.basename(db)

        if self.db.endswith(".db"):
            self.db = self.db[:-3]

        self.coverage = cov
        self.path_bin = path_bin
        self.path_scripts = path_scripts

        if self.db.endswith(".db"):
            self.db = self.db[:-3]

        for strLine in open(self.db + ".db"):
            strLine = strLine.strip()

            if strLine.startswith("blocks"):
                self.set_blocks( int( strLine[ strLine.find("=")+1 : ].strip() ) )
                break

    def replace_variables(self, strCmd, **args):
        return strCmd.format(db_path = self.db_path,
                             coverage = self.coverage,
                             db = self.db,
                             path = self.path_bin,
                             path_scripts = self.path_scripts,
                             **args)

    def command_to_threads(self, cmd, threads):

        if "daligner" in cmd:
            args = shlex.split(cmd)

            threads = queue.DEFAULT_DALIGNER_THREADS

            for i in range(len(args)):
                arg = args[i]

                if arg.startswith("-j"):
                    if len(arg) > 2:
                        threads = int( arg[2:].strip() )
                    else:
                        threads = int( args[i + 1] )

                    break

        return threads
