from pid import PID
from particle import Particle
from process import Process

from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
import matplotlib.pyplot as plt
import threading
import numpy as np


class Plotter(object):
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
        plt.legend(handles=self.handles, bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
        plt.xlabel('Time $sec$')
        plt.ylabel('Position $m$')


        axamp = plt.axes([0.25, 0.15, 0.65, 0.03])
        self.slider = Slider(axamp, 'Target', -1, 1, valinit=0)
        self.slider.on_changed(self.update_target)

        self.fig.canvas.mpl_connect('close_event', self.stop)

    def start(self):
        self.threads = []
        for process in self.processes:
            thread = threading.Thread(target=process.infinity_loop, args=[self.dt])
            thread.start()
            self.threads.append(thread)
        
        self.ani = FuncAnimation(self.fig, self.update, frames=np.arange(1000), blit=True, interval=1)
        plt.show()
    
    def update_target(self, event):
        target = self.slider.val
        for process in processes:
            process.set_target(target)

    def stop(self, event):
        for process in self.processes:
            process.stop()
        for thread in self.threads:
            thread.join()

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
        
        for idx, process in enumerate(self.processes):
            result = process.result()
            if idx == 0:
                self.handles[idx].set_data(result['t'][:480], result['y'][-480:])
            self.handles[idx+1].set_data(result['t'][:480], result['x'][-480:])
        
        return self.handles


if __name__ == '__main__':
    pid_params = [
        dict(kp=0.4, ki=0, kd=0),
        dict(kp=1.5, ki=0., kd=0.5),
        dict(kp=0.2, ki=0.1, kd=0.01),
    ]
    processes = []
    for c in pid_params:
        processes.append(Process(particle=Particle(x0=[0], v0=[0], inv_mass=1.), pid=PID(**c)))

    plotter = Plotter(processes)
    plotter.start()