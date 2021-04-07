from threading import current_thread
import numpy as np
from pid import PID
from particle import Particle
import time
import sys

class Process(object):
    
    real_time = True

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
        self._target = 1.


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
            if Process.real_time:
                time.sleep(dt/2)


    def stop(self):
        self._run = False
        
        


class TunedProcess(Process):
    def __init__(self, particle, pid, batch_size):
        super().__init__(particle=particle, pid=pid)
        self.batch_count = 0
        self.batch_size = batch_size
    
    
    def correct_pid(self):
        pass
    

    def step(self, dt=0.01):
        super().step(dt=dt)
        
        self.batch_count += 1
        if self.batch_count == self.batch_size:
            self.correct_pid()
            self.batch_count = 0

        return self.result()


class TwiddleTunedProcess(TunedProcess):
    def __init__(self, particle=Particle(), pid=PID(kp=0.0, ki=0.0, kd=0.0), batch_size=50):
        super().__init__(particle=particle,pid=pid, batch_size=batch_size)
        self.err = sys.float_info.max / 2
        self.params = dict(
            kp=self.pid.kp, 
            ki=self.pid.ki, 
            kd=self.pid.kd)
        self.dparams = dict((key, 1.) for (key, _) in self.params.items())
        self.keys_count = 0

    def _cost_func(self):
        return np.sum(np.square(self.result()['e'][-self.batch_size:]))/ min(self.batch_size, self.result()['e'].size)

    def correct_pid(self):
        k = list(self.params.keys())[self.keys_count]
        self.keys_count += 1
        self.keys_count = self.keys_count % len(self.params.keys())

        self.params[k] += self.dparams[k]
        e = self._cost_func()

        if e < self.err:
            self.err = e
            self.dparams[k] *= 1.1
        else:
            self.params[k] -= self.dparams[k]
            e = self._cost_func()

            if e < self.err:
                self.err = e
                self.dparams[k] *= 1.1
            else:
                self.params[k] += self.dparams[k]
                self.dparams[k] *= 0.95
       
        self.pid.kp = self.params['kp']
        self.pid.ki = self.params['ki']
        self.pid.kd = self.params['kd']


class GradiendBasedProcess(TunedProcess):
    def __init__(self, particle, pid=PID()):
        super().__init__(particle, pid, batch_size=1)
        self.differential = np.asarray([0.])
        self.alfa = 1
        self.params = np.asarray((self.pid.kp, self.pid.ki, self.pid.kd))
        # self.prior_x = np.asarray([self.pid.kp, self.pid.ki, self.pid.kd])

    def update(self, dt=0.01):
        e = super().update(dt=dt)
        if self.result() != None:
            current_dif = (e - self.result()['e'][-1]) / dt
            self.differential = np.append(self.differential, np.asanyarray(current_dif))
        return e
    
    def correct_pid(self):
        self.params = self.params - (self.differential[-1] * self.alfa)
        self.pid.kp = self.params[0]
        self.pid.ki = self.params[1]
        self.pid.kd = self.params[2]
        print(self.params)