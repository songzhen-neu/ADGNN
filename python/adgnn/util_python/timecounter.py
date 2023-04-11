import time
import numpy as np
from adgnn.context import context

class TimeCounter():
    def __init__(self):
        self.time_list = {}
        self.start_time = {}

    # def clear_time(self):
    #     for id in context.glContext.time_epoch.keys():
    #         context.glContext.time_epoch[id] = 0

    def start(self, key):
        self.start_time[key] = time.time()

    def end(self, key):
        end = time.time()
        if not self.time_list.__contains__(key):
            self.time_list[key] = []
        self.time_list[key].append(end - self.start_time[key])

    def show(self, key):
        print((key + ' time: {:.4f}').format(self.time_list[key][-1]))

    def printAvrgTime(self):
        for id in self.time_list.keys():
            print('average ' + str(id) + ' time: {:.4f}s'.format(np.array(self.time_list[id]).mean()))

    def printTotalTime(self):
        for id in self.time_list.keys():
            print('total ' + str(id) + ' time: {:.4f}s'.format(np.array(self.time_list[id]).sum()))

time_counter = TimeCounter()
