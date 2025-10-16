import numpy as np
import math
from GenZ.unit import Unit
import json
class System(object):
    compute_multiplier = {'int8': 0.5, 'bf16': 1, 'f32': 2, 'int4': 0.25, 'int2':0.125, 'fp8': 0.5,  'fp6':0.5, 'fp4': 0.25}
    mem_multiplier = {'int8': 1, 'bf16': 2, 'f32': 4, 'int4':0.5, 'int2':0.25, 'fp8':1,  'fp6':0.75, 'fp4':0.5}
    def __init__(self, unit=None,
                flops=123, mxu_shape=None,
                onchip_mem_bw=18000, on_chip_mem_size=float('Inf'),
                offchip_mem_bw=900, off_chip_mem_size=float('Inf'),
                external_mem_bw=0,
                frequency=940, bits='bf16',
                compute_efficiency=1, memory_efficiency=1, comm_efficiency=1,
                interchip_link_bw = 25, num_nodes = 1, interchip_link_latency=1.9,
                compute_engine='GenZ',    # GenZ or Scale-sim
                collective_strategy='GenZ',    # GenZ or ASTRA-SIM
                topology='FullyConnected',
                parallelism_heirarchy = "TP{1}_EP{1}_PP{1}",
                network_config = None,
                gear_params = None,
                # ---- 新增：L2缓存参数 ----
                l2_cache_size=128,            # MB - L2缓存容量
                l2_cache_bw=2000,             # GB/s - L2缓存带宽（固定）
                ):

        if unit is None:
            self.unit = Unit()
        else:
            self.unit = unit

        self.flops = self.unit.unit_to_raw(flops, type='C')
        self.op_per_sec = self.flops/2

        self.frequency = self.unit.unit_to_raw(frequency, type='F')
        self.onchip_mem_bw = self.unit.unit_to_raw(onchip_mem_bw, type='BW')
        self.offchip_mem_bw = self.unit.unit_to_raw(offchip_mem_bw, type='BW')
        self.interchip_link_bw = self.unit.unit_to_raw(interchip_link_bw, type='BW')
        self.interchip_link_latency = interchip_link_latency * 1e-6     ## us
        self.external_mem_bw = self.unit.unit_to_raw(external_mem_bw, type='BW')
        self.on_chip_mem_size = self.unit.unit_to_raw(on_chip_mem_size, type='M')
        self.on_chip_mem_left_size = self.unit.unit_to_raw(on_chip_mem_size, type='M')
        self.off_chip_mem_size = self.unit.unit_to_raw(off_chip_mem_size, type='M')
        
        # ---- 新增：L2缓存初始化 ----
        self.l2_cache_size = self.unit.unit_to_raw(l2_cache_size, type='M')
        self.l2_cache_bw = self.unit.unit_to_raw(l2_cache_bw, type='BW')
        self.l2_cache_used = 0  # 当前L2缓存使用量
        self.compute_efficiency = compute_efficiency
        self.memory_efficiency = memory_efficiency
        self.comm_efficiency = comm_efficiency
        self.mxu_shape = mxu_shape

        self.compute_engine = compute_engine
        assert self.compute_engine in ['GenZ', 'Scale-sim'], "Invalid compute_engine. Must be one of: GenZ, Scale-sim"

        self.collective_strategy = collective_strategy
        assert self.collective_strategy in ['GenZ', 'ASTRA-SIM'], "Invalid collective_strategy. Must be one of: GenZ, ASTRA-SIM"
        self.num_nodes = num_nodes
        self.topology = topology
        self.bits = bits
        self.parallelism_heirarchy = parallelism_heirarchy   ## TP{1}_EP{1}_PP{1}
        self.network_config = network_config
        if gear_params:
            self.gear_r = gear_params['r']
            self.gear_s = gear_params['s']
            self.gear_b = gear_params['b']
            self.quantization_type = 'gear'
        else:
            self.quantization_type = None

    def __str__(self):
        unit = Unit()
        a = f"Accelerator OPS: {unit.raw_to_unit(self.flops,type='C')} TOPS , Freq = {unit.raw_to_unit(self.frequency,type='F')} GHz, Num Nodes = {self.num_nodes} \n"
        b = f"On-Chip mem size: {unit.raw_to_unit(self.on_chip_mem_size, type='M')} MB , Off-chip mem size:{unit.raw_to_unit(self.off_chip_mem_size, type='M')} MB\n"
        c = f"On-Chip mem BW: {unit.raw_to_unit(self.onchip_mem_bw, type='BW')} GB/s , Off-chip mem BW:{unit.raw_to_unit(self.offchip_mem_bw, type='BW')} GB/s, External-mem BW:{unit.raw_to_unit(self.external_mem_bw, type='BW')} GB/s,\n"
        return a+b+c

    def get_params(self):
        unit = Unit()
        a = f"Accelerator OPS: {unit.raw_to_unit(self.flops,type='C')} TOPS , Freq = {unit.raw_to_unit(self.frequency,type='F')} GHz, Num Nodes = {self.num_nodes}"
        b = f" Off-chip mem size:{unit.raw_to_unit(self.off_chip_mem_size, type='M')/1024} GB "
        c = f" Off-chip mem BW:{unit.raw_to_unit(self.offchip_mem_bw, type='BW')} GB/s, External-mem BW:{unit.raw_to_unit(self.external_mem_bw, type='BW')} GB/s"
        return a+b+c

    @classmethod
    def from_dict(cls, config_dict):
        init_params = cls.__init__.__code__.co_varnames[1:cls.__init__.__code__.co_argcount]
        filtered_params = {k: v for k, v in config_dict.items() if k in init_params}
        return cls(**filtered_params)

    @classmethod
    def from_json(cls, json_str):
        config_dict = json.loads(json_str)
        return cls.from_dict(config_dict)

    def set_onchip_mem_bw(self,onchip_mem_bw):
        self.onchip_mem_bw = self.unit.unit_to_raw(onchip_mem_bw, type='BW')

    def set_offchip_mem_bw(self,offchip_mem_bw):
        self.offchip_mem_bw = self.unit.unit_to_raw(offchip_mem_bw, type='BW')

    def get_offchip_mem_bw(self):
        return self.unit.raw_to_unit(self.offchip_mem_bw,type='BW')

    def get_external_mem_bw(self):
        return self.unit.raw_to_unit(self.external_mem_bw,type='BW')

    def get_interchip_link_bw(self):
        return self.unit.raw_to_unit(self.interchip_link_bw,type='BW')

    def get_off_chip_mem_size(self):
        return self.unit.raw_to_unit(self.off_chip_mem_size,type='M')


    def claim_onchip_mem(self, data_sz):
        if data_sz > self.on_chip_mem_left_size:
            raise ValueError(f'Not enough on-chip memory: Need {data_sz}, only has {self.on_chip_mem_size}')
        self.on_chip_mem_left_size -= data_sz
        return self.on_chip_mem_left_size

    def release_onchip_mem(self, data_sz):
        self.on_chip_mem_left_size = max(self.on_chip_mem_size, data_sz + self.on_chip_mem_left_size)
        return self.on_chip_mem_left_size

    def get_bit_multiplier(self, type='C', data='w', operators=None):
        if type == 'C':
            return self.compute_multiplier[self.bits]
        elif type == 'M':
            if self.quantization_type == 'gear':
                if data == 'k' or data == 'v':
                    # print(
                    #     f"Quantized KV bits: {self.mem_multiplier[self.gear_b]}",
                    #     f"Sparsity bits:{(self.gear_s/100) * self.mem_multiplier[self.bits]}",
                    #     f"Operators: ", operators,
                    #     f"Low Rank Bits: {((np.prod(operators[:-2])/np.prod(operators)) * (operators[-2]*self.gear_r + operators[-1]*self.gear_r) * self.mem_multiplier[self.bits])}")

                    return (    self.mem_multiplier[self.gear_b]
                                + (self.gear_s/100) * self.mem_multiplier[self.bits]
                                + ((np.prod(operators[:-2])/np.prod(operators)) * (operators[-2]*self.gear_r + operators[-1]*self.gear_r) * self.mem_multiplier[self.bits])
                    )
            return self.mem_multiplier[self.bits]
    
    # ---- 新增：L2缓存管理方法 ----
    def claim_l2_cache(self, data_sz):
        """申请L2缓存空间"""
        if data_sz > self.l2_cache_size - self.l2_cache_used:
            return False  # L2缓存不足
        self.l2_cache_used += data_sz
        return True

    def release_l2_cache(self, data_sz):
        """释放L2缓存空间"""
        self.l2_cache_used = max(0, self.l2_cache_used - data_sz)
        return self.l2_cache_used

    def get_l2_cache_hit_rate(self, data_sz, op_type='GEMM', access_pattern='sequential', phase='decode', seq_len=1):
        """计算L2缓存命中率 - 基于算子类型、访问模式和数据重用模式"""
        
        # 基于算子类型的缓存友好性
        cache_friendly_ops = {
            'GEMM': 0.7,      # 矩阵乘法有较好的空间局部性
            'Attend': 0.4,    # Attention机制访问模式复杂
            'FFN': 0.6,       # 前馈网络中等局部性
            'Embedding': 0.1,  # 嵌入层随机访问
            'Norm': 0.8,      # 归一化层顺序访问
        }
        
        base_hit_rate = cache_friendly_ops.get(op_type, 0.5)
        
        # ---- 修复：改进容量因子计算 ----
        # 考虑长序列时数据量激增对缓存容量的压力
        available_cache = max(0, self.l2_cache_size - self.l2_cache_used)
        
        # 对于长序列，数据量通常与序列长度平方相关（特别是Attention）
        if op_type == 'Attend' and seq_len > 1000:
            # Attention的复杂度是O(n^2)，数据量增长更快
            effective_data_sz = data_sz * (seq_len / 1000) ** 0.5  # 平方根增长
        elif op_type in ['GEMM', 'FFN'] and seq_len > 1000:
            # 矩阵运算数据量线性增长
            effective_data_sz = data_sz * (seq_len / 1000) ** 0.3  # 较慢增长
        else:
            effective_data_sz = data_sz
            
        capacity_factor = min(1.0, available_cache / effective_data_sz)
        
        # 基于访问模式的影响
        if access_pattern == 'sequential':
            pattern_factor = 1.0
        elif access_pattern == 'random':
            pattern_factor = 0.3
        else:  # 'mixed'
            pattern_factor = 0.6
        
        # ---- 修复：更保守的数据重用模式考虑 ----
        reuse_boost = 0.0
        
        # 1. 权重数据重用（在prefill阶段权重被大量重用）
        if phase == 'prefill' and op_type in ['GEMM', 'FFN']:
            # 权重数据在prefill阶段被多次访问，但长序列时重用效果递减
            if seq_len <= 1000:
                reuse_boost += 0.15
            elif seq_len <= 2000:
                reuse_boost += 0.10  # 长序列时重用效果降低
            else:
                reuse_boost += 0.05  # 超长序列时重用效果进一步降低
        
        # 2. 激活数据的时间局部性（长序列时局部性反而变差）
        if phase == 'prefill':
            if seq_len <= 1000:
                # 中等长度序列有较好的局部性
                if op_type == 'Attend':
                    reuse_boost += 0.08
                elif op_type in ['GEMM', 'FFN']:
                    reuse_boost += 0.05
            elif seq_len <= 2000:
                # 长序列时局部性开始变差
                if op_type == 'Attend':
                    reuse_boost += 0.03  # 大幅降低
                elif op_type in ['GEMM', 'FFN']:
                    reuse_boost += 0.02
            else:
                # 超长序列时局部性很差，几乎不给重用加成
                reuse_boost += 0.01
        
        # 3. 基于算子类型的数据重用特性（更保守）
        if op_type == 'Attend' and phase == 'prefill':
            # Attention机制在prefill阶段的重用模式，长序列时效果递减
            if seq_len <= 1000:
                reuse_boost += 0.08
            elif seq_len <= 2000:
                reuse_boost += 0.04
            else:
                reuse_boost += 0.01
        
        # ---- 修改：基于数据大小的惩罚因子 ----
        # 根据访存数据大小计算惩罚，而不是序列长度
        data_size_penalty = 1.0
        
        # 将数据大小转换为MB
        data_size_mb = self.unit.raw_to_unit(effective_data_sz, type='M')
        
        if data_size_mb > 1000:
            # 超大数据量时显著降低命中率
            data_size_penalty = 0.01
        elif data_size_mb > 500:
            # 超大数据量时显著降低命中率
            data_size_penalty = 0.05
        elif data_size_mb > 200:
            # 超大数据量时显著降低命中率
            data_size_penalty = 0.1
        elif data_size_mb > 100:
            # 大数据量时适度降低命中率
            data_size_penalty = 0.2
        elif data_size_mb > 50:
            # 中等数据量时轻微降低命中率
            data_size_penalty = 0.3
        elif data_size_mb > 20:
            # 小数据量时轻微降低命中率
            data_size_penalty = 0.4
        
        # 综合计算命中率
        hit_rate = (base_hit_rate + reuse_boost) * capacity_factor * pattern_factor * data_size_penalty
        final_hit_rate = min(1.0, max(0.0, hit_rate))
        
        # ---- DEBUG: 添加调试信息 ----
        if data_sz > self.unit.unit_to_raw(20, type='M'):  # 只对大数据量打印
            print(f"🔍 L2缓存调试: {op_type} | seq_len={seq_len} | phase={phase} | data_sz={self.unit.raw_to_unit(data_sz, type='M'):.1f}MB")
            print(f"   base={base_hit_rate:.3f} | reuse={reuse_boost:.3f} | capacity={capacity_factor:.3f} | pattern={pattern_factor:.3f} | data_penalty={data_size_penalty:.3f}")
            print(f"   effective_data_sz={self.unit.raw_to_unit(effective_data_sz, type='M'):.1f}MB | available_cache={self.unit.raw_to_unit(available_cache, type='M'):.1f}MB")
            print(f"   final_hit_rate={final_hit_rate:.3f}")
            print()
        
        return final_hit_rate

    def get_l2_cache_bw(self):
        """获取L2缓存带宽"""
        return self.unit.raw_to_unit(self.l2_cache_bw, type='BW')