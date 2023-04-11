from adgnn.sample.sample import Sample
import numpy as np
import torch
import sys
from adgnn.util_python.timecounter import time_counter
from adgnn.context import context
import psutil,os
from adgnn.util_python.data_trans import transGraphCppToPython

class RandomSample(Sample):
    def sample(self, fan_out, epoch,  **kwargs):
        time_counter.start("randomSample")
        context.glContext.sample.randomSample(fan_out)
        time_counter.end("randomSample")
        time_counter.start("transGraphCppToPython-sample")
        transGraphCppToPython("train", "sample")
        time_counter.end("transGraphCppToPython-sample")
        return context.glContext.graph_sample

random_sample = RandomSample()
