# 动态导入通信后端模块
_comm_backend = None

def set_comm_backend(backend_type):
    """设置通信后端类型
    
    Args:
        backend_type (str): 'nv_like' 或 'pcie_like'
    """
    global _comm_backend
    if backend_type == 'nv_like':
        from GenZ.collective_times_nvlike import get_AR_time, get_AG_time, get_A2A_time, get_message_pass_time
    elif backend_type == 'pcie_like':
        from GenZ.collective_times_pcielike import get_AR_time, get_AG_time, get_A2A_time, get_message_pass_time
    else:
        raise ValueError(f"不支持的通信后端类型: {backend_type}。支持的类型: 'nv_like', 'pcie_like'")
    
    _comm_backend = {
        'get_AR_time': get_AR_time,
        'get_AG_time': get_AG_time,
        'get_A2A_time': get_A2A_time,
        'get_message_pass_time': get_message_pass_time
    }

def get_AR_time(data, numNodes, system):
    """get_AR_time

    Args:
        data (int): Message size(Bytes) per node to complete all reduce.
        num_AR_nodes (int): Number of nodes among which all-reduce is performed
        system (System object): Object of class System

    Returns:
        time(float): Total time(msec) to complete the All-Reduce
    """
    if _comm_backend is None:
        # 默认使用nv_like后端
        set_comm_backend('nv_like')
    
    return _comm_backend['get_AR_time'](data, numNodes, system)

def get_AG_time(data, numNodes, system):
    """get_AG_time

    Args:
        data (int): Message size(Bytes) per node to complete all gather.
        num_AG_nodes (int): Number of nodes among which all-gather is performed
        system (System object): Object of class System

    Returns:
        time(float): Total time(msec) to complete the All-Gather
    """
    if _comm_backend is None:
        # 默认使用nv_like后端
        set_comm_backend('nv_like')
    
    return _comm_backend['get_AG_time'](data, numNodes, system)

def get_message_pass_time(data, system):
    """get_message_pass_time

    Args:
        data (int): Message size(Bytes) per node to pass from 1 decide to next.
        system (System object): Object of class System

    Returns:
        time(float): Total time(msec) to pass the Message from 1 node to next
    """
    if _comm_backend is None:
        # 默认使用nv_like后端
        set_comm_backend('nv_like')
    
    return _comm_backend['get_message_pass_time'](data, system)


def get_A2A_time(data, numNodes, system):
    """get_A2A_time

    Args:
        data (int): Total message size (Bytes) per node to be exchanged in all-to-all.
        num_A2A_nodes (int): Number of nodes participating in the all-to-all operation.
        system (System object): Object of class System

    Returns:
        time (float): Total time (msec) to complete the All-to-All operation
    """
    if _comm_backend is None:
        # 默认使用nv_like后端
        set_comm_backend('nv_like')
    
    return _comm_backend['get_A2A_time'](data, numNodes, system)