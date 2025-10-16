import numpy as np
from operator import mul
from math import ceil
from GenZ.unit import Unit
from GenZ.system import System
from GenZ.Models import OpType, CollectiveType
from GenZ.collective_times import get_AR_time, get_A2A_time, get_message_pass_time, get_AG_time
import re
# 4, 5 Regular Logit and Attend
# 9, 10 Beam Merge Logit and attend
op_type_dicts = {0: 'FC', 1: 'CONV2D', 2: 'DWCONV', 3: 'GEMM', 4: 'Logit', 5: 'Attend', 6:'Sync',
                9:'Logit', 10:'Attend', 11:'CONV1D', 12:'Einsum', 13:'Repeat', 14:'EndRepeat',
                15:'Norm', 16:'Avg', 17:'Special_Func'}
class Operator(object):
    def __init__(self, dim, density=(1.0,1.0,1.0)):
        self.dim = [int(x) if isinstance(x, (int, float, np.int32, np.int64)) else x for x in dim]
        self.density_a, self.density_w, self.density_o = density
        self.input_a, self.input_w, self.output = self.get_tensors()
        self.num_ops = self.get_num_ops()
        self.set_mem_pin(*self.get_default_mem_loc())

    def get_default_mem_loc(self):
        return ['off', 'off', 'off']

    def set_mem_pin(self, input_a=None, input_b=None, output=None):
        if input_a is not None:
            self.input_a_loc = input_a
        if input_b is not None:
            self.input_w_loc = input_b
        if output is not None:
            self.output_loc = output

    def set_tensor(self, input_a=None, input_w=None, output=None):
        if input_a is not None:
            self.input_a = input_a
        if input_w is not None:
            self.input_w = input_w
        if output is not None:
            self.output = output

    def get_density_list(self):
        return [self.density_a, self.density_w, self.density_o]

    def get_op_type(self, dim):
        return op_type_dicts[dim[-1]]

    def _get_access_pattern(self, op_type, data_size):
        """根据算子类型和数据大小推断访问模式"""
        if op_type in ['GEMM', 'FC']:
            # 矩阵运算通常是顺序访问
            return 'sequential'
        elif op_type in ['Attend', 'Logit']:
            # Attention机制有复杂的访问模式
            return 'mixed'
        elif op_type in ['Embedding']:
            # 嵌入层通常是随机访问
            return 'random'
        elif op_type in ['Norm', 'Avg']:
            # 归一化层通常是顺序访问
            return 'sequential'
        else:
            # 默认混合模式
            return 'mixed'
    
    def _infer_computation_context(self, system):
        """推断计算阶段和序列长度"""
        # 默认值
        phase = 'decode'
        seq_len = 1
        
        # 基于算子类型推断计算阶段
        op_type = self.get_op_type(self.dim)
        
        # 获取数据大小信息来推断序列长度
        sz_list = self.get_operators_size(system)
        if len(sz_list) > 0:
            # 使用第一个张量的大小来推断序列长度
            first_tensor_sz = sz_list[0]
            
            # ---- 修复：更准确的序列长度推断 ----
            if op_type == 'Attend':
                # Attention算子的输入张量大小通常与序列长度平方相关
                if first_tensor_sz > system.unit.unit_to_raw(200, type='M'):  # 200MB
                    seq_len = 3000  # 超长序列
                    phase = 'prefill'
                elif first_tensor_sz > system.unit.unit_to_raw(100, type='M'):  # 100MB
                    seq_len = 2000  # 长序列
                    phase = 'prefill'
                elif first_tensor_sz > system.unit.unit_to_raw(50, type='M'):  # 50MB
                    seq_len = 1000  # 中等序列
                    phase = 'prefill'
                else:
                    seq_len = 100   # 短序列
                    phase = 'decode'
            elif op_type in ['GEMM', 'FFN']:
                # 矩阵运算的序列长度推断（线性增长）
                if first_tensor_sz > system.unit.unit_to_raw(300, type='M'):  # 300MB
                    seq_len = 3000
                    phase = 'prefill'
                elif first_tensor_sz > system.unit.unit_to_raw(200, type='M'):  # 200MB
                    seq_len = 2000
                    phase = 'prefill'
                elif first_tensor_sz > system.unit.unit_to_raw(100, type='M'):  # 100MB
                    seq_len = 1000
                    phase = 'prefill'
                else:
                    seq_len = 1
                    phase = 'decode'
            elif op_type == 'Logit':
                # Logit算子通常在decode阶段
                phase = 'decode'
                seq_len = 1
            
            # ---- DEBUG: 添加调试信息 ----
            if first_tensor_sz > system.unit.unit_to_raw(50, type='M'):  # 只对大数据量打印
                print(f"🔍 上下文推断: {op_type} | tensor_sz={system.unit.raw_to_unit(first_tensor_sz, type='M'):.1f}MB | 推断结果: phase={phase}, seq_len={seq_len}")
        
        return phase, seq_len

    def get_tensors(self):
        pass

    def get_size(self, tensor):
        return np.prod(tensor)

    # Each kind of operation function will have its own num ops, in which using the layer parameters obtained from the
    # .csv file it will give out number of required ops .
    def get_num_ops(self):
        pass

    def get_dimensions(self):
        return self.get_tensors()

    # For each kind of operator, this returns number of required paramters for that layer type. (Refer operators.py )
    def get_effective_dim_len(self):
        pass

    def get_num_data(self):
        return sum(self.get_sz_list())

    def get_effective_num_data(self, system):
        return sum(self.get_operators_size(system))


    def get_ideal_memory_time(self, system):
        sz_list = self.get_sz_list()
        memory_time_onchip = 0
        memory_time_offchip = 0
        for tensor_sz in sz_list:
            memory_time_onchip += tensor_sz * system.get_bit_multiplier(type='M')/ system.onchip_mem_bw
            memory_time_offchip += tensor_sz * system.get_bit_multiplier(type='M')/ system.offchip_mem_bw
        return  memory_time_offchip, memory_time_onchip


    def get_compute_time(self, system):
        if system.compute_engine == 'GenZ':
            return self.get_effective_num_ops(system) * system.get_bit_multiplier(type='C')/system.op_per_sec
        elif system.compute_engine == 'Scale-sim':
            from .Scale_Sim.get_scale_sim_time import get_scale_sim_time
            return get_scale_sim_time(op=self, system=system)


    def get_effective_num_ops(self, system=None):
        return self.get_num_ops()


# The function returns the size of each of the 3 models parameter for each layer, i.e. input, weights and outputs.
    def get_sz_list(self):
        return list(map(self.get_size, [self.input_a, self.input_w, self.output]))

    def get_loc_list(self):
        return [self.input_a_loc, self.input_w_loc, self.output_loc]

    def get_operators_size(self, system):
        sz_list = self.get_sz_list()
        operators_sizes = []
        for i, tensor_sz in enumerate(sz_list):
            if self.get_op_type(self.dim) in ['Logit', 'Attend']:
                if i == 1 and self.get_op_type(self.dim) == 'Logit':
                    ## K values
                    operators_sizes.append(tensor_sz * system.get_bit_multiplier(type='M', data='k', operators=self.input_w))
                elif i == 1 and self.get_op_type(self.dim) == 'Attend':
                    ## V values
                    operators_sizes.append(tensor_sz * system.get_bit_multiplier(type='M', data='v', operators=self.input_w))
                else:
                    operators_sizes.append(tensor_sz * system.get_bit_multiplier(type='M', data='a'))
            else:
                operators_sizes.append(tensor_sz * system.get_bit_multiplier(type='M', data='w'))

        return operators_sizes

    def get_memory_time(self, system):
        sz_list = self.get_operators_size(system)
        loc_list = self.get_loc_list()
        memory_time = 0
        
        # ---- 修改：引入智能L2缓存建模 ----
        op_type = self.get_op_type(self.dim)
        
        # ---- 新增：推断计算阶段和序列长度 ----
        phase, seq_len = self._infer_computation_context(system)
        
        for tensor_sz, loc in zip(sz_list, loc_list):
            if loc == 'off':
                # 片外内存访问：基于算子类型、访问模式和数据重用模式计算命中率
                access_pattern = self._get_access_pattern(op_type, tensor_sz)
                hit_rate = system.get_l2_cache_hit_rate(tensor_sz, op_type, access_pattern, phase, seq_len)
                
                if hit_rate > 0:
                    # 部分数据在L2缓存中
                    cached_data = tensor_sz * hit_rate
                    uncached_data = tensor_sz * (1 - hit_rate)
                    l2_time = cached_data / system.l2_cache_bw  # L2缓存访问
                    offchip_time = uncached_data / system.offchip_mem_bw  # 片外内存访问
                    memory_time += l2_time + offchip_time
                    
                    # ---- DEBUG: 添加内存访问调试信息 ----
                    if tensor_sz > system.unit.unit_to_raw(50, type='M'):  # 只对大数据量打印
                        print(f"🔍 内存访问: {op_type} | tensor_sz={system.unit.raw_to_unit(tensor_sz, type='M'):.1f}MB | hit_rate={hit_rate:.3f}")
                        print(f"   cached={system.unit.raw_to_unit(cached_data, type='M'):.1f}MB | uncached={system.unit.raw_to_unit(uncached_data, type='M'):.1f}MB")
                        print(f"   l2_time={l2_time*1000:.3f}ms | offchip_time={offchip_time*1000:.3f}ms | total={memory_time*1000:.3f}ms")
                        print()
                else:
                    # 完全缓存未命中
                    offchip_time = tensor_sz / system.offchip_mem_bw
                    memory_time += offchip_time
                    
                    # ---- DEBUG: 添加缓存未命中调试信息 ----
                    if tensor_sz > system.unit.unit_to_raw(50, type='M'):  # 只对大数据量打印
                        print(f"🔍 缓存未命中: {op_type} | tensor_sz={system.unit.raw_to_unit(tensor_sz, type='M'):.1f}MB | offchip_time={offchip_time*1000:.3f}ms")
                        print()
            elif loc == 'on':
                # 片上内存访问（假设已经在L1缓存中）
                bw = system.onchip_mem_bw
                memory_time += tensor_sz / bw
            else:
                raise ValueError(f'Wrong bw allocation: {loc}.')
        return memory_time

    def get_communication_time(self, system):
        '''
            Returns the communication time for the operator in seconds.
        '''
        if self.get_op_type(self.dim) != 'Sync':
            return 0
        else:
            data_size = self.communication_data() * system.get_bit_multiplier(type='M', data='a')
            if system.collective_strategy == 'GenZ':
                if self.collective_type == CollectiveType.AllReduce:
                    return get_AR_time(data_size , self.num_collective_nodes, system) / 1000
                elif  self.collective_type == CollectiveType.All2All:
                    return get_A2A_time(data_size , self.num_collective_nodes, system) / 1000
                elif  self.collective_type == CollectiveType.MessagePass:
                    return get_message_pass_time(data_size, system) / 1000
                elif self.collective_type == CollectiveType.AllGather:
                    return get_AG_time(data_size, self.num_collective_nodes, system) / 1000
                else:
                    raise ValueError(f'Unknown collective type: {self.collective_type}.')
            elif system.collective_strategy == 'ASTRA-SIM':
                from .Astra_sim.get_astra_sim_time import get_astrasim_collective_time, get_network_config, merge_parallelism_heirarchy
                "ALLREDUCE", "ALLTOALL", "ALLGATHER", "REDUCESCATTER"
                collective_convertion = { CollectiveType.AllReduce: 'ALLREDUCE', CollectiveType.All2All: 'ALLTOALL',
                                CollectiveType.AllGather: 'ALLGATHER', CollectiveType.ReduceScatter: 'REDUCESCATTER', 
                                }
                if system.network_config is None:
                    return max(get_astrasim_collective_time(system=system, collective_type=collective_convertion[self.collective_type],
                                                        collective_size=data_size).values())/1e9
                else:
                    parallelism_heirarchy = system.parallelism_heirarchy
                    if self.collective_type == CollectiveType.MessagePass:
                        parallelism = "PP"
                    elif self.collective_type == CollectiveType.AllReduce:
                        TP_nodes = int(re.search(r'TP\{(\d+)\}', parallelism_heirarchy).group(1))
                        if self.num_collective_nodes != TP_nodes:
                            # Only EP dimension is used as TP dimension
                            parallelism_heirarchy = merge_parallelism_heirarchy(parallelism_heirarchy, merge_dim='EP', merge_into='TP')
                        parallelism = "TP"
                    elif self.collective_type == CollectiveType.All2All:
                        parallelism = "EP"
                    elif self.collective_type == CollectiveType.AllGather:
                        TP_nodes = int(re.search(r'TP\{(\d+)\}', parallelism_heirarchy).group(1))
                        if self.num_collective_nodes != TP_nodes:
                            # Only EP dimension is used as TP dimension
                            parallelism_heirarchy = merge_parallelism_heirarchy(parallelism_heirarchy, merge_dim='EP', merge_into='TP')
                        parallelism = "TP"
                    else:
                        raise ValueError(f'Unknown parallelism for collective type: {self.collective_type}.')

                    network_config = get_network_config(network_config = system.network_config, 
                                                        parallelism_heirarchy = parallelism_heirarchy,
                                                        parallelism = parallelism)
                    if parallelism == "PP":
                            BW = network_config['bandwidth'][0]
                            lat = network_config['latency'][0]
                            # TODO : There is a bug with astrasim when num_nodes = 2
                            # Using GenZ for now.
                            temp_sys = System(num_nodes=2, topology='FullyConnected', interchip_link_bw=BW, interchip_link_latency=lat)
                            # pipe_time = max(get_astrasim_collective_time(system=temp_sys,
                            #                                     collective_type="ALLTOALL",
                            #                                     collective_size=data_size/2).values())/1e6
                            pipe_time = get_message_pass_time(data_size, temp_sys) / 1000
                            if len(network_config['npus_count']) > 1:   ## PP over more than 1 dimension, we need average time.
                                # Num hops: (dim[0]-1)*dim[1] + dim[1]-1
                                first_dim_time = pipe_time * (network_config['npus_count'][0]-1) * network_config['npus_count'][1] 
                                temp_sys = System(num_nodes=2, topology='FullyConnected',
                                                interchip_link_bw=network_config['bandwidth'][1], interchip_link_latency=network_config['latency'][1]) 
                                # second_dim_time = max(get_astrasim_collective_time(system=temp_sys,
                                #                             collective_type="ALLTOALL",
                                #                             collective_size=data_size/2).values())/1e6 * (network_config['npus_count'][1]-1)
                                second_dim_time = (get_message_pass_time(data_size, temp_sys) / 1000) * (network_config['npus_count'][1]-1)
                                return (first_dim_time + second_dim_time)/(network_config['npus_count'][0] * network_config['npus_count'][1] -1)
                            else:
                                return pipe_time
                    else:
                        return max(get_astrasim_collective_time(collective_type=collective_convertion[self.collective_type],
                                                        collective_size=data_size, network_config=network_config).values())/1e9

    def get_onchip_occupancy(self):
        sz_list = self.get_sz_list()
        loc_list = self.get_loc_list()
        onchip_mem_occupancy = 0
        for tensor_sz, loc in zip(sz_list, loc_list):
            if loc == 'on':
                onchip_mem_occupancy += tensor_sz

        return onchip_mem_occupancy

    def get_model_characterstics(self, system, unit = Unit()):
        num_ops =  self.get_num_ops()
        num_data = self.get_effective_num_data(system)
        op_intensity = num_ops/num_data  if num_data else 0
        input_a_size, input_w_size, output_size = self.get_operators_size(system)
        ret = {
            'Layer Name': self.name,
            'Op Type': self.get_op_type(self.dim),
            'Dimension': self.get_dimensions(),
            'Op Intensity': op_intensity,
            f'Num ops ({unit.unit_flop})': unit.raw_to_unit(num_ops, type='O'),
            f'Input_a ({unit.unit_mem})': unit.raw_to_unit(input_a_size, type='M'),
            f'Input_w ({unit.unit_mem})': unit.raw_to_unit(input_w_size, type='M'),
            f'Output ({unit.unit_mem})': unit.raw_to_unit(output_size, type='M'),
            f'Total Data ({unit.unit_mem})': unit.raw_to_unit(self.get_effective_num_data(system), type='M'),
        }

        return ret

    def get_roofline(self, system, unit):
        ideal_complete_offchip_time, ideal_complete_onchip_time = self.get_ideal_memory_time(system=system)
        # x2 for ops -> MAC has 1 multiplication and 1 Addition hence 2.
        num_ops = self.get_effective_num_ops(system) * 2
        num_data = self.get_effective_num_data(system)
        op_intensity = num_ops/num_data if num_data else 0

        compute_time = self.get_compute_time(system=system)

        compute_time /= system.compute_efficiency
        compute_efficiency = system.compute_efficiency

        memory_time = self.get_memory_time(system=system) / system.memory_efficiency

        comm_time = self.get_communication_time(system=system) / system.comm_efficiency

        ## This is special case when there is no calculations
        if compute_time == 0:
            memory_time = 0
        exec_time = max(compute_time, memory_time, comm_time)
        thrpt = num_ops/exec_time if exec_time else 0
        com_to_mem_ratio = compute_time/memory_time if memory_time else 0
        if com_to_mem_ratio == 0:
            boundedness = 'Collective'
        else:
            boundedness = 'Compute' if com_to_mem_ratio > 1 else 'Memory'

        input_a_size, input_w_size, output_size = self.get_operators_size(system)

        if exec_time != 0:
            compute_util, memory_util, comm_util = compute_time/exec_time, memory_time/exec_time, comm_time/exec_time
        else:
            compute_util, memory_util, comm_util = 0, 0, 0

        # ---- 新增：计算L2缓存命中率统计 ----
        total_l2_hit_rate = 0
        l2_cache_impact = 0
        
        # 获取计算上下文信息
        phase, seq_len = self._infer_computation_context(system)
        op_type = self.get_op_type(self.dim)
        
        for tensor_sz, loc in zip(self.get_operators_size(system), self.get_loc_list()):
            if loc == 'off':
                access_pattern = self._get_access_pattern(op_type, tensor_sz)
                hit_rate = system.get_l2_cache_hit_rate(tensor_sz, op_type, access_pattern, phase, seq_len)
                total_l2_hit_rate += hit_rate * tensor_sz
                l2_cache_impact += tensor_sz
        
        avg_l2_hit_rate = total_l2_hit_rate / l2_cache_impact if l2_cache_impact > 0 else 0

        ret = {
            'Layer Name': self.name,
            'Op Type': self.get_op_type(self.dim),
            'Dimension': self.get_dimensions(),
            'Bound': boundedness,
            'C/M ratio': com_to_mem_ratio,
            'Op Intensity': op_intensity,
            f'Latency ({unit.unit_time})': unit.raw_to_unit(exec_time, type='T'),
            f'Cycles': exec_time*system.frequency,
            f'C Effcy': compute_efficiency,
            f'Num ops ({unit.unit_flop})': unit.raw_to_unit(num_ops, type='O'),
            f'Input_a ({unit.unit_mem})': unit.raw_to_unit(input_a_size, type='M'),
            f'Input_w ({unit.unit_mem})': unit.raw_to_unit(input_w_size, type='M'),
            f'Output ({unit.unit_mem})': unit.raw_to_unit(output_size, type='M'),
            f'Total Data ({unit.unit_mem})': unit.raw_to_unit(self.get_effective_num_data(system), type='M'),
            f'Throughput ({unit.unit_compute})': unit.raw_to_unit(thrpt, type='C'),
            f'Compute time ({unit.unit_time})': unit.raw_to_unit(compute_time, type='T'),
            f'Memory time ({unit.unit_time})': unit.raw_to_unit(memory_time, type='T'),
            f'Communication time ({unit.unit_time})': unit.raw_to_unit(comm_time, type='T'),
            f'Compute cycle': compute_time*system.frequency,
            f'Memory cycle': memory_time*system.frequency,
            f'Communication cycle': comm_time*system.frequency,
            f'Compute Utilization': compute_util,
            f'Memory Utilization': memory_util,
            f'Communication Utilization': comm_util,
            f'L2 Cache Hit Rate': avg_l2_hit_rate,
            f'L2 Cache Size ({unit.unit_mem})': unit.raw_to_unit(system.l2_cache_size, type='M'),
            f'L2 Cache Used ({unit.unit_mem})': unit.raw_to_unit(system.l2_cache_used, type='M'),
        }

        return ret










