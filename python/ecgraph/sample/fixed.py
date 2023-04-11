from ecgraph.sample.sample import Sample
from ecgraph.context import context
from ecgraph.util_python.data_trans import transGraphCppToPython

class FixedSample(Sample):
    def sample(self,graph, fan_out, epoch, batch_size,  **kwargs):
        if epoch==0:
            context.glContext.sample.randomSample(fan_out)
            transGraphCppToPython("train", "sample")
        return context.glContext.graph_sample


fixed_sample = FixedSample()
