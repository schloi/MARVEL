
from __future__ import print_function

import sys
import os
import subprocess
import time
import shlex
import glob
import logging

import marvel.config as mconfig

try:
    from subprocess import DEVNULL
except ImportError:
    DEVNULL = open(os.devnull, 'wb')

class rawqueue(object):

    QUEUE_POLL = 0.5

    def __init__(self, cores, poll = QUEUE_POLL, blocks = None, checkpoint = None, finish_block_on_error = False):
        self.queue = []
        self.disable_add = False

        self.parallel = cores
        self.checkpoint = checkpoint
        self.blocks = blocks
        self.poll = poll

        self.finish_block_on_error = finish_block_on_error

    def get_blocks(self):
        return self.blocks

    def set_blocks(self, blocks):
        self.blocks = blocks

    def ignore(self, bEnable = True):
        if bEnable and self.disable_add:
            logging.warn("ignore already active")
        elif not bEnable and not self.disable_add:
            logging.warn("ignore already disable")

        self.disable_add = bEnable

    def resume(self):
        self.queue = []

    def replace_variables(self, strCmd, **args):
        return strCmd.format(**args)

    def assign_block_arguments(self, block, args):
        bargs = {}

        for (k, v) in args.items():
            if type(v) in ( type( () ), type([]) ):
                bargs[k] = v[block - 1]
            else:
                bargs[k] = v

        return bargs

    def block(self, strCmd, first = 1, last = -1, threads = 1, **args):
        if self.disable_add:
            return None

        if self.blocks is None:
            logging.error("WARNING: number of blocks not set. ignoring block() call.")
            return None

        if last == -1:
            last = self.blocks + 1
        else:
            last = min(last + 1, self.blocks + 1)

        arrTask = []
        for nBlock in range(first, last):
            bargs = self.assign_block_arguments(nBlock, args)

            arrTask.append( ( self.replace_variables(strCmd, threads = threads, block = nBlock, **bargs), threads ) )
        self.queue.append( arrTask )

        return len(self.queue) - 1

    def single(self, strCmd, threads = 1, **args):
        if self.disable_add:
            return None

        self.queue.append( [ ( self.replace_variables(strCmd, threads = threads, **args), threads ) ] )

        return len(self.queue) - 1

    def merge(self, steps = 1):
        if self.disable_add:
            return None

        while steps > 0:
            if len(self.queue) < 2:
                return

            self.queue[-2].extend( self.queue[-1] )
            del self.queue[-1]

            steps -= 1

    def expand_dump(self, cmd):
        result = []

        expand = cmd.find("[expand:")

        if expand != -1:
            left = expand + len("[expand:")
            right = cmd.find("]", left)
            cmd = cmd[:expand] + cmd[left:right] + cmd[right + 1:]

        return cmd

    def expand(self, arr):
        result = []

        for item in arr:
            if item.startswith("[expand:"):
                strPattern = item[ item.find(":")+1 : -1 ]
                result += glob.glob(strPattern)
            else:
                result.append(item)

        return result

    def command_to_threads(self, cmd, threads):
        return threads

    def plan(self, plan, first = 1, last = -1, threads = 1, path = mconfig.PATH_BIN):
        if isinstance(plan, str):
            plan = self.replace_variables(plan)
            commands = open(plan).readlines()
        else:
            commands = plan.readlines()

        arrBlock = []

        if first < 1 or last != -1 and last < first:
            logging.error("WARNING: loading plan with illegal first {} and last {}. reverting to defaults.".format(first, last))
            first = 1
            last = -1

        for strLine in commands:
            if strLine[0] == '#':
                if len(arrBlock) > 0:
                    self.queue.append(arrBlock)
                    arrBlock = []

                if first != 1 or last != -1:
                    logging.error("WARNING: loading plan with multiple parallel command sets, but first and last specified. reverting to defaults.")
                    first = 1
                    last = -1

                continue

            strCmd = strLine.strip()

            threads = self.command_to_threads(strCmd, threads)

            if path != None and strCmd[0] != os.path.sep:
                strCmd = os.path.join(path, strCmd)

            arrBlock.append( (strCmd, threads) )

        if len(arrBlock) > 0:
            if first > len(arrBlock) or last != -1 and last > len(arrBlock):
                logging.error("WARNING: loading plan with illegal first {} and last {}. reverting to defaults.".format(first, last))
                first = 1
                last = -1

            if last != -1:
                arrBlock = arrBlock[:last]

            if first != 1:
                arrBlock = arrBlock[first - 1:]

            self.queue.append( arrBlock )

    def write_checkpoint(self, level, qstatus):
        fileOut = open(self.checkpoint, "w")

        if False not in qstatus:
            task = len(qstatus)
        else:
            task = qstatus.index(False)

        fileOut.write("{} {}".format(level, task))
        fileOut.close()

    def delete_checkpoint(self):
        os.remove(self.checkpoint)

    def read_checkpoint(self):
        fileIn = open(self.checkpoint, "r")
        line = fileIn.readline().strip()
        items = line.split()

        level = 0
        task = 0

        if len(items) == 2:
            try:
                level = int(items[0])
                task = int(items[1])
            except:
                pass

        fileIn.close()

        return (level, task)

    def dump(self, fileOut):
        for i in range( len(self.queue) ):

            for command in self.queue[i]:
                cmd = self.expand_dump(command[0])

                fileOut.write(cmd + "\n")

            if i < len(self.queue) - 1:
                fileOut.write("# jobs prior must be complete in order to continue\n")

    def process(self, resume_from_checkpoint = False):
        if len(self.queue) == 0:
            return

        arrProcesses = []

        nLevel = 0
        nTask = 0
        nThreads = 0
        nExit = False
        qstatus = [False] * len(self.queue[nLevel])

        if resume_from_checkpoint:
            if not os.path.exists(self.checkpoint):
                logging.error("WARNING: resume from checkpoint requested, but no checkpoint found")
            else:
                (nLevel, nTask) = self.read_checkpoint()

                logging.info("RESUME {} {}".format(nLevel, nTask))

                qstatus = [True] * nTask + [False] * (len(self.queue[nLevel]) - nTask)

        while nLevel < len(self.queue):
            if len(arrProcesses) > 0:
                time.sleep( self.poll )

            for i in range(len(arrProcesses) - 1, -1, -1):
                (proc, strCmd, threads, taskid) = arrProcesses[i]

                if proc.poll() != None:
                    if proc.returncode != 0:
                        logging.info("exit {0} {1}".format(proc.returncode, strCmd))

                        if not self.finish_block_on_error:
                            sys.exit(1)
                        else:
                            nExit = True
                    else:
                        (process_output, dummy) = proc.communicate()

                        logging.info(process_output)

                        qstatus[taskid] = True

                        if self.checkpoint != None:
                            self.write_checkpoint(nLevel, qstatus)

                    del arrProcesses[i]
                    nThreads -= threads

            if nTask == len(self.queue[nLevel]):
                if len(arrProcesses) > 0:
                    continue
                else:
                    nLevel += 1
                    nTask = 0
                    nThreads = 0

                    if nExit:
                        break

                    if nLevel == len(self.queue):
                        break

                    qstatus = [False] * len(self.queue[nLevel])

                    # print("processing {0} of {1}".format(nLevel + 1, len(self.queue)))

            while nTask != len(self.queue[nLevel]) and nThreads < self.parallel:
                (strCmd, threads) = self.queue[nLevel][nTask]

                if len(self.queue[nLevel]) > 1:
                    print("[{} {} | {} {}] {}".format(nLevel + 1, len(self.queue), nTask + 1, len(self.queue[nLevel]), strCmd))
                else:
                    print("[{} {}] {}".format(nLevel + 1, len(self.queue), strCmd))

                logging.info("[{} {} | {} {}] {}\n".format(nLevel + 1, len(self.queue), nTask + 1, len(self.queue[nLevel]), strCmd))

                arrCmd = shlex.split(strCmd)
                arrCmd = self.expand(arrCmd)

                if strCmd[0] == '!':
                    arrCmd[0] = arrCmd[0][1:]

                    arrProcesses.append( (subprocess.Popen( strCmd[1:],
                                                            bufsize = -1,
                                                            stdout = DEVNULL, stderr = subprocess.STDOUT,
                                                            shell = True),
                                          strCmd, threads, nTask) )
                else:
                    if not os.path.exists(arrCmd[0]):
                        logging.warning("command {} not found. potentially using non-absolute paths".format(arrCmd[0]))

                    arrProcesses.append( (subprocess.Popen( arrCmd,
                                                            bufsize = -1,
                                                            stdout = DEVNULL, stderr = subprocess.STDOUT),
                                          strCmd, threads, nTask) )

                nTask += 1
                nThreads += threads

        self.queue = []
        self.disable_add = False

        if self.checkpoint != None:
            self.delete_checkpoint()
