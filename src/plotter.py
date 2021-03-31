from pid import PID
from particle import Particle
from process import Process, TwiddleTunedProcess

from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
import matplotlib.pyplot as plt
import threading
import numpy as np
import random
import sys


class Plotter(object):
    
    allow_random = True

    def __init__(self, processes=[Process()], dt=0.02):
        self.processes = processes
        self.handles = []
        self.dt = dt
        
        fig_params = dict(num=None, figsize=(10, 5), dpi=100, facecolor='w', edgecolor='k')
        
        self.fig, self.ax = plt.subplots(**fig_params)
        
        self.ax.set_ylim(-2, 3)
        self.ax.set_xlim(0, dt*500)
        
        self.init_draw()

        plt.title('Particle trajectory')
        plt.subplots_adjust(right=0.7, bottom=0.3)
        plt.legend(handles=self.handles, bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        plt.xlabel('Time $sec$')
        plt.ylabel('Position $m$')


        axamp = plt.axes([0.25, 0.15, 0.65, 0.03])
        self.slider = Slider(axamp, 'Target', -1.5, 1.5, valinit=1)
        self.slider.on_changed(self.update_target)

        self.fig.canvas.mpl_connect('close_event', self.stop)

    def start_ani(self):
        self.ani = FuncAnimation(self.fig, self.update, frames=np.arange(1000), blit=True, interval=1)
        plt.show()
    
    def start_processes(self):
        self.threads = []
        for process in self.processes:
            thread = threading.Thread(target=process.infinity_loop, args=[self.dt])
            thread.start()
            self.threads.append(thread)

    def start(self):
        self.start_processes()
        self.start_ani()
    
    def update_target(self, event):
        self._set_target(self.slider.val)
            
    def _set_target(self, target):
        for process in self.processes:
            process.set_target(target)
    
    def random_target(self, chanse=0.003):
        if random.random() <= chanse:
            self._set_target(random.uniform(-1.5, 1.5))

    def stop_processes(self):
        for process in self.processes:
            process.stop()
        for thread in self.threads:
            thread.join()
    
    def stop(self, event):
        self.stop_processes()

    def init_draw(self):
        
        for idx, process in enumerate(self.processes):
            result = process.step()
            if idx == 0:
                fh, = plt.plot(result['t'], result['y'], label='target')
                self.handles.append(fh)    

            xh, = plt.plot(result['t'], result['x'], label='pid kp {:.2f} ki {:.2f} kd {:.2f}'.format(process.pid.kp, process.pid.ki, process.pid.kd))
            self.handles.append(xh)
        
        return self.handles

    def update(self, t):
        if Plotter.allow_random:
            self.random_target()
        
        for idx, process in enumerate(self.processes):
            result = process.result()
            if idx == 0:
                self.handles[idx].set_data(result['t'][:480], result['y'][-480:])
            self.handles[idx+1].set_data(result['t'][:480], result['x'][-480:])
        
        return self.handles


class AbstractGenesicPlotter(Plotter):
    
    random_range = dict(a=0, b=10)

    def __init__(self, dt=0.02, population=100, preview=5, batch_size=50, particle=Particle(x0=[0], v0=[0], inv_mass=1.)):
        self.particle = particle
        self.preview = preview
        self.batch_size = batch_size
        self.batch_count = 0
        self.population = population
        self.all_processes = [self.gen_random_process() for _ in range(self.population)]
        super().__init__(processes=self.all_processes[:self.preview], dt=dt)
    
    def start_processes(self):
        self.threads = []
        for process in self.processes:
            thread = threading.Thread(target=process.infinity_loop, args=[self.dt])
            thread.start()
            self.threads.append(thread)

    def _get_pid_params(self, procees):
        return dict(
            kp=procees.pid.kp,
            ki=procees.pid.ki, 
            kd=procees.pid.kd)

    def _random_pid_params(self):
        return  dict(
            kp=random.uniform(**AbstractGenesicPlotter.random_range),
            ki=random.uniform(**AbstractGenesicPlotter.random_range), 
            kd=random.uniform(**AbstractGenesicPlotter.random_range))
            
    def gen_random_process(self):
        
        return Process(particle=self.particle, pid=PID(**self._random_pid_params()))

    def _cost_func(self, process, fix_size=0.00000001):
        return  (np.sum(np.square(process.result()['e'][-self.batch_size:]))) / min(self.batch_size, process.result()['e'].size, 1) + fix_size
    
    def _calc_probability(self, processes):
        processes_probability = []
        for process in processes:
            processes_probability.append((self._cost_func(process), process))
        return self._norm(processes_probability)
    
    def _norm(self, processes_probability):
        summ = sum(c[0] for c in processes_probability)
        func = 0
        distribution = {}

        for val in processes_probability:
            func += val[0] / summ
            distribution[func] = val[1]

        return distribution
    
    def _crosssover(self, distribution):
        pass

    def _reproduction(self, distribution, crossover_count=None):
        
        new_population = []
        
        if crossover_count == None:
            crossover_count = int(self.population / 3)
        
        for _ in range(crossover_count):
            new_population.append(self._crosssover(distribution))
        return new_population
        
    def _mutate(self, process, mutatuion_prop=0.01):
        pass

    def _mutations(self, processes,  mutation_count=None):
        if mutation_count == None:
            mutation_count = int(self.population / 10)
        for _ in range(mutation_count):
            process = random.choice(processes) 
            mutate  = self._mutate(process)
            processes[processes.index(processes)] = mutate
        return processes
            

    def _gen_new_generation(self, processes):
        distribution = self._calc_probability(processes)
        print('-------')
        for process in processes:
            print(self._cost_func(process))
        print('-------')
        print(distribution)
        print('-------')
        processes.extend(self._reproduction(distribution))
        processes = self._mutations(processes)
        processes = sorted(processes, key = lambda process: self._cost_func(process))
        
        
        return processes[:self.population]

    def _recalc_generation(self):
        self.stop_processes()
        self.all_processes = self._gen_new_generation(self.all_processes)
        self.start_processes()


    def update(self, t):
        if (self.batch_count == self.batch_size):
            self._recalc_generation()
            self.batch_count = 0
        self.batch_count += 1
        return super().update(t)


if __name__ == '__main__':
    pid_params = [
        # dict(kp=0.4, ki=0, kd=0),
        dict(kp=1.5, ki=0., kd=0.5),
        # dict(kp=0.2, ki=0.1, kd=0.01),
    ]
    processes = [TwiddleTunedProcess(particle=Particle(x0=[0], v0=[0], inv_mass=1.))]
    for c in pid_params:
        processes.append(Process(particle=Particle(x0=[0], v0=[0], inv_mass=1.), pid=PID(**c)))

    plotter = Plotter(processes)
    plotter.start()