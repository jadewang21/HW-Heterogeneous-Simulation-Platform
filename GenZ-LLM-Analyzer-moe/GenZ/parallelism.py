import numpy as np

class ParallelismConfig():
    r"""
    This is the configuration class to store the configuration of a Model Splitting.
    It is used to instantiate an LLM into multiple parallel units
    according to the specified arguments, defining the degree of various parallelism.
    Args:

    """
    def __init__(
        self,
        tensor_parallel=1,
        pipeline_parallel=1,
        data_parallel=1,
        expert_parallel=1,
        sequence_parallel=1,
        ep_share_tp_group=False,
        **kwargs,
    ):
        self.tensor_parallel = tensor_parallel
        self.pipeline_parallel = pipeline_parallel
        self.data_parallel = data_parallel
        self.expert_parallel = expert_parallel
        self.sequence_parallel = sequence_parallel
        # 当为 True 且 ep==tp 时，表示 MoE 专家并行与 TP 共享同一组，
        # 专家本身不在 TP 维度上切分；专家内不进行 FFN AR。
        self.ep_share_tp_group = ep_share_tp_group
        self.total_chips = np.prod([
                            self.data_parallel,
                            self.expert_parallel,
                            self.sequence_parallel,
                            self.pipeline_parallel,
                            self.tensor_parallel])

        super().__init__(**kwargs)

    def __str__(self):
        return str(vars(self))