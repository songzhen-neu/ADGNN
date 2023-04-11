from adgnn.sample.sample import Sample
from adgnn.context import context
from adgnn.util_python.data_trans import transGraphCppToPython

class FixedSample(Sample):
    def sample(self,graph, fan_out, epoch, batch_size,  **kwargs):
        if epoch==0:
            context.glContext.sample.randomSample(fan_out)
            transGraphCppToPython("train", "sample")
        return context.glContext.graph_sample


fixed_sample = FixedSample()
