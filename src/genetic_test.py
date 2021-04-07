from copy import copy
from pid import PID
from plotter import AbstractGenesicPlotter
import random
import sys
import threading
import copy

from process import Process

class MyGenesicPlotter(AbstractGenesicPlotter):
    def _find(self, keys, nearest_key):
        low = 0
        high = len(keys) - 1
        mid = (high + low) // 2
        while low <= high:
            mid = (high + low) // 2
            if keys[mid] < nearest_key:
                low = mid + 1
            else:
                high = mid - 1
        return keys[mid] 
    
    def _crosssover(self, distribution):
        keys = list(distribution.keys())
        parents = [distribution[self._find(keys, random.random())] for _ in range(2)]
        parants_pid = [self._get_pid_params(parent) for parent in parents]
        child_pid_params = dict((key, random.choice([parant_pid[key] for parant_pid in parants_pid])) for (key, _ ) in parants_pid[0].items())
        # child = copy.copy(parents[0])
        child = self.gen_random_process()
        child.pid = PID(**child_pid_params)
        return child
    
    def _mutate(self, process, mutatuion_prop=0.01, sigma=1):
        if random.random() < mutatuion_prop:
            pid_params = self._get_pid_params(process)
            pid_params = dict((key,  random.gauss(val, sigma)) for (key, val) in pid_params.items())
            process.pid = PID(pid_params)
        return process
            

if __name__ == '__main__':

    plotter = MyGenesicPlotter(dt=0.02, population=6, preview=1, batch_size=100)
    plotter.start()
    