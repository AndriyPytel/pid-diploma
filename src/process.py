import numpy as np
import math
from pid import PID
from particle import Particle
import time

class Process(object):
    
    def __init__(self, particle=Particle(), pid=PID()):
        self.particle = particle
        self.pid = pid
        
        self.reset()

    
    def set_target(self, t):
        self._target = t


    def target(self):
        return np.asarray([self._target])
                

    def sense(self):
        return self.particle.x
    

    def correct(self, error, dt):
        return self.pid.update(error, dt)


    def actuate(self, u, dt):
        self.particle.add_force(u)
        self.particle.update(dt)


    def update(self, dt=0.01):
        self.y = self.target()
        self.x = self.sense()
        self.e = self.y - self.x
        self.u = self.correct(self.e, dt)

        self.actuate(self.u, dt)

        self.t += dt

        return self.e


    def reset(self):
        self.t = 0.
        self._result = None
        self._target = 0.


    def step(self, dt=0.01):
        self.update(dt)

        if self._result == None: 
            self.fields = [
                    ('y', np.float32, self.y.shape),
                    ('x', np.float32, self.x.shape), 
                    ('e', np.float32, self.e.shape),
                    ('u', np.float32, self.u.shape),
                    ('t', np.float32, 1)            
                ]
            self._result = np.asarray((self.y, self.x, self.e, self.u, self.t - dt), dtype=self.fields)
        else:
            self._result = np.append(self._result, np.asarray((self.y, self.x, self.e, self.u, self.t - dt), dtype=self.fields))   
    
        return self._result

    def result(self):
        return self._result

    def infinity_loop(self, dt=0.01):
        self._run = True

        while self._run:
            self.step(dt)
            time.sleep(dt/2)


    def stop(self):
        self._run = False
