
import os
import shlex
import logging
import sys
import subprocess
import time

try:
    from subprocess import DEVNULL
except ImportError:
    DEVNULL = open(os.devnull, 'wb')

import marvel.config

class gpuqueue:
    def __init__(self, devices : int, path_bin : str = marvel.config.PATH_BIN):
        self.path_bin = path_bin
        self.devices = devices
        self.work = []

    def plan(self, plan):
        if isinstance(plan, str):
            commands = open(plan).readlines()
        else:
            commands = plan.readlines()

        for cmd in commands:
            if self.path_bin != None and cmd[0] != os.path.sep:
                cmd = os.path.join(self.path_bin, cmd)

            cmd = cmd.strip()
            self.work.append( cmd )

    def process(self):
        if len(self.work) == 0:
            return

        devices = [None] * self.devices
        task = 0

        while task < len(self.work) or devices.count(None) != self.devices:
            if len(devices) > 0:
                time.sleep( 1 )

            for i in range(self.devices):
                if devices[i] == None:
                    continue

                (proc, cmd, taskid) = devices[i]

                if proc.poll() != None:
                    if proc.returncode != 0:
                        logging.info("exit {0} {1}".format(proc.returncode, cmd))
                        sys.exit(1)
                    else:
                        (process_output, dummy) = proc.communicate()
                        logging.info(process_output)

                    devices[i] = None

            if task < len(self.work):
                for i in range(self.devices):
                    if devices[i] != None:
                        continue

                    cmd = self.work[task]

                    arrCmd = shlex.split(cmd)
                    arrCmd.insert(1, "-g{}".format(i))

                    print("[{} {}] {}".format(task + 1, len(self.work), " ".join(arrCmd)))

                    if not os.path.exists(arrCmd[0]):
                        logging.warning("command {} not found. potentially using non-absolute paths".format(arrCmd[0]))

                    devices[i] = ( (subprocess.Popen( arrCmd, bufsize = -1, stdout = DEVNULL, stderr = subprocess.STDOUT),
                                    cmd,
                                    task) )

                    task += 1

                    if task == len(self.work):
                        break

        self.queue = []
