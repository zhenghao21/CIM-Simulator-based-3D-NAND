import numpy as np
def multi_dim_index(base_indices, dim_sizes, target_dim, base_idx, delta):
    """
    计算多维数组中带有借位/进位的索引
    
    参数:
    base_indices: 原始索引列表 [i, j, k, ...]
    dim_sizes: 各维度大小列表 [dim1, dim2, dim3, ...]
    target_dim: 目标操作的维度（从0开始）
    base_idx: 目标维度的基础索引值
    delta: 要添加的偏移量
    
    返回:
    新的完整索引列表
    """
    if base_indices[0]<0:
        print('索引超出范围')
    # 创建新索引列表的副本
    new_indices = base_indices.copy()
    
    # 计算目标维度的新索引值
    new_idx = base_idx+delta
    
    # 处理借位/进位
    # 当前维度
    current_dim = target_dim
    
    # 处理借位情况
    if new_idx < 0:
        # 向前一维度借位
        new_idx+=dim_sizes[current_dim]
        new_indices[current_dim]=new_idx
        if new_indices[current_dim]>=0:
            current_dim-=1
            new_indices=multi_dim_index(new_indices, dim_sizes, current_dim, new_indices[current_dim], -1)
        else:
            new_indices[current_dim-1]-=1
            new_indices=multi_dim_index(new_indices, dim_sizes, current_dim, new_indices[current_dim],0)
        return new_indices
    # 处理进位情况
    elif new_idx >= dim_sizes[current_dim]:
        # 向前一维度进位
        new_idx -= dim_sizes[current_dim]
        new_indices[current_dim] = new_idx
        if current_dim > 0:  # 确保还有前一维度可以进位
            current_dim -= 1
            new_indices = multi_dim_index(new_indices, dim_sizes, current_dim, new_indices[current_dim], 1)
        else:
            print('索引超出最高维度范围')
        return new_indices
    else:
        new_indices[current_dim]=new_idx
        return new_indices
def transform_index(shape, index, delta):
    """
    优化版本：直接在多维索引上进行计算，避免完整的扁平化
    
    参数:
    shape: tuple，数组的形状
    index: tuple，当前索引
    delta: int，变化量
    
    返回:
    tuple，变换后的索引
    """
    # 转换为列表以便修改
    new_index = list(index)
    
    # 从最低维度开始处理
    carry = delta
    for i in range(len(shape) - 1, -1, -1):
        new_index[i] += carry
        
        # 处理进位和退位
        if new_index[i] >= 0:
            carry = new_index[i] // shape[i]
            new_index[i] = new_index[i] % shape[i]
        else:
            # 处理负数情况
            carry = -((-new_index[i] - 1) // shape[i] + 1)
            new_index[i] = new_index[i] % shape[i]
    
    # 检查是否超出边界
    if carry != 0:
        raise ValueError("索引超出数组范围")
    
    return tuple(new_index)
def create_multidim_list(dimensions, fill_value=None):
    """
    创建任意维度的列表
    :param dimensions: 一个包含每个维度大小的列表，如 [2, 3, 4] 创建 2x3x4 的三维列表
    :param fill_value: 填充值，默认为None
    :return: 多维列表
    """
    if not dimensions:
        return fill_value
    return [create_multidim_list(dimensions[1:], fill_value) for _ in range(dimensions[0])]

def unpackbits_extended(x, count=None, axis=-1, bitorder='little'):
    """
    扩展版的 np.unpackbits 函数，可以处理 uint8、uint16、uint32 等类型的数据
    
    参数:
    - x: 需要展开的数组，可以是 uint8, uint16, uint32 等类型
    - count: 要解包的位数，None 表示所有位
    - axis: 在哪个轴上展开位
    - bitorder: 'big' 或 'little'，表示位顺序
    
    返回:
    - 展开后的位数组，类型为 int8
    """
    # 获取输入数据类型的最大位数
    dtype = x.dtype
    if np.issubdtype(dtype, np.uint8):
        total_bits = 8
    elif np.issubdtype(dtype, np.uint16):
        total_bits = 16
    elif np.issubdtype(dtype, np.uint32):
        total_bits = 32
    elif np.issubdtype(dtype, np.uint64):
        total_bits = 64
    else:
        raise TypeError(f"Unsupported dtype: {dtype}, expected uint8, uint16, uint32 or uint64")
    
    # 如果没有指定count，使用数据类型的最大位数
    if count is None:
        count = total_bits
    elif count > total_bits:
        raise ValueError(f"count ({count}) cannot be greater than the number of bits in dtype ({total_bits})")
    
    # 对于uint8类型，可以直接使用numpy的unpackbits
    if np.issubdtype(dtype, np.uint8) and count <= 8:
        return np.unpackbits(x, count=count, axis=axis, bitorder=bitorder).astype(np.int8)
    
    # 处理uint16及更高位的类型
    # 我们使用一个按位操作来提取每一位，并将其存储到结果数组中
    
    # 确保axis为正数
    if axis < 0:
        axis = x.ndim + axis
        
    # 创建一个新的形状，在axis处扩展一个维度
    out_shape = list(x.shape)
    out_shape.insert(axis + 1, count)
    
    # 创建结果数组
    out = np.zeros(out_shape, dtype=np.int8)
    
    # 确定展开哪些位
    if bitorder == 'little':
        bit_indices = list(range(count))
    else:  # 'big'
        bit_indices = list(range(total_bits - count, total_bits))
    
    # 将输入数据reshape为扁平化的形式，便于处理
    x_flat = x.reshape(-1)
    
    # 计算每个位元素的索引偏移
    offsets = np.arange(len(x_flat))
    
    # 对每个位进行展开
    for i, bit in enumerate(bit_indices):
        # 创建用于索引的切片列表
        idx = [slice(None)] * (x.ndim + 1)
        idx[axis + 1] = i
        
        # 计算当前位的值
        bit_val = ((x_flat & (1 << bit)) > 0).astype(np.int8)
        
        # 重塑回原始维度，并添加位维度
        bit_val_reshaped = np.zeros(out_shape, dtype=np.int8)
        bit_val_reshaped_flat = bit_val_reshaped.reshape(-1)
        
        # 相当于每个count位重复一次原始数据的每个元素
        step = np.prod(out_shape[axis+2:]) if axis+2 < len(out_shape) else 1
        for j in range(len(x_flat)):
            indices = offsets[j] * count * step + i * step + np.arange(step)
            if indices.size > 0:
                bit_val_reshaped_flat[indices] = bit_val[j]
        
        # 将重塑后的位值赋给输出数组
        out[tuple(idx)] = bit_val_reshaped.reshape(out_shape)[tuple(idx)]
    
    return out
    
def quantize_per_tensor(tensor, bits, signed=True, per_channel=False, channel_dim=-1, symmetric=True):
    """
    将浮点tensor量化到指定位宽，支持有符号(补码)和无符号量化，可以选择逐通道或逐张量量化
    
    参数:
    - tensor: 输入的浮点数张量
    - bits: 量化位宽
    - signed: 是否为有符号量化，默认为True
    - per_channel: 是否逐通道量化，默认为False (逐张量)
    - channel_dim: 通道维度，默认为最后一个维度(-1)
    - symmetric: 是否为对称量化，默认为True，False表示使用非对称量化
    
    返回:
    - quantized_tensor: 量化后的整数张量
    - scale: 量化缩放因子，如果per_channel为True，则为每个通道的缩放因子数组
    - zero_point: 零点偏移，如果symmetric为False则返回，否则为None或None的数组
    """
    if signed:
        # 有符号补码的范围
        quant_min = -(1 << (bits - 1))
        quant_max = (1 << (bits - 1)) - 1
    else:
        # 无符号的范围
        quant_min = 0
        quant_max = (1 << bits) - 1
    
    # 初始化量化后的张量
    dtype = np.int8 if signed and bits <= 8 else np.uint8 if bits <= 8 else np.int16 if signed else np.uint16
    quantized_tensor = np.zeros_like(tensor, dtype=dtype)
    
    if per_channel:
        # 逐通道量化
        # 计算通道个数
        num_channels = tensor.shape[channel_dim]
        
        # 初始化scale数组和zero_point数组(如果使用非对称量化)
        scale = np.zeros(num_channels, dtype=np.float32)
        zero_point = np.zeros(num_channels, dtype=dtype) if not symmetric else None
        
        # 对每个通道进行量化
        for i in range(num_channels):
            # 创建索引来选择当前通道
            indices = [slice(None)] * tensor.ndim
            indices[channel_dim] = i
            
            # 获取当前通道的数据
            channel_data = tensor[tuple(indices)]
            
            if symmetric:
                # 对称量化
                channel_max = np.max(np.abs(channel_data))
                if channel_max > 0:
                    scale[i] = channel_max / max(abs(quant_min), abs(quant_max))
                else:
                    scale[i] = 1.0
                
                # 量化当前通道: 缩放并裁剪到量化范围
                quantized_float = np.clip(channel_data / scale[i], quant_min, quant_max)
                quantized_tensor[tuple(indices)] = np.round(quantized_float).astype(dtype)
            
            else:
                # 非对称量化
                channel_min = np.min(channel_data)
                channel_max = np.max(channel_data)
                
                # 如果通道数据是常量，设置一个默认缩放值
                if channel_min == channel_max:
                    scale[i] = 1.0
                    zero_point[i] = 0
                else:
                    # 计算缩放因子
                    scale[i] = (channel_max - channel_min) / (quant_max - quant_min)
                    
                    # 计算零点偏移
                    zero_point_float = quant_min - channel_min / scale[i]
                    # 确保zero_point在量化范围内
                    zero_point[i] = np.clip(round(zero_point_float), quant_min, quant_max).astype(dtype)
                
                # 量化公式: q = round(x / scale) + zero_point
                quantized_float = channel_data / scale[i] + zero_point[i]
                quantized_tensor[tuple(indices)] = np.clip(np.round(quantized_float), quant_min, quant_max).astype(dtype)
    
    else:
        # 逐张量量化
        if symmetric:
            # 对称量化 (原有逻辑)
            tensor_max = np.max(np.abs(tensor))
            scale = tensor_max / max(abs(quant_min), abs(quant_max)) if tensor_max > 0 else 1.0
            zero_point = None
            
            # 量化: 缩放并裁剪到量化范围
            quantized_float = np.clip(tensor / scale, quant_min, quant_max)
            quantized_tensor = np.round(quantized_float).astype(dtype)
        
        else:
            # 非对称量化
            tensor_min = np.min(tensor)
            tensor_max = np.max(tensor)
            
            if tensor_min == tensor_max:
                scale = 1.0
                zero_point = 0
            else:
                # 计算缩放因子
                scale = (tensor_max - tensor_min) / (quant_max - quant_min)
                
                # 计算零点偏移
                zero_point_float = quant_min - tensor_min / scale
                # 确保zero_point在量化范围内
                zero_point = np.clip(round(zero_point_float), quant_min, quant_max).astype(dtype)
            
            # 量化公式: q = round(x / scale) + zero_point
            quantized_float = tensor / scale + zero_point
            quantized_tensor = np.clip(np.round(quantized_float), quant_min, quant_max).astype(dtype)
    if symmetric:
        return quantized_tensor, scale,None
    else:
        return quantized_tensor, scale,zero_point


def pad_to_target_shape(array, target_size, axis):
    """
    将NumPy数组的指定维度扩展到target_size，并用0填充

    参数:
    - array: 输入数组
    - target_size: 目标尺寸
    - axis: 要扩展的维度

    返回:
    - padded_array: 扩展并填充后的数组
    """
    # 计算需要填充的大小
    current_size = array.shape[axis]
    if current_size >= target_size:
        print("无需填充")
        return array

    # 计算在指定维度上需要填充的前后大小
    pad_width = [(0, 0)] * array.ndim
    pad_width[axis] = (0, target_size - current_size)

    # 使用np.pad进行填充
    padded_array = np.pad(array, pad_width=pad_width, mode='constant', constant_values=0)
    return padded_array
def split_by_labels(data, labels, prefixes):
    """
    通用的数据分组函数，按照标签前缀提取数据
    
    参数:
    - data: 数据数组
    - labels: 标签数组，格式如"Wq0", "Wk1"等
    - prefixes: 要提取的标签前缀列表，如['Wq', 'Wk', 'Wv']
    
    返回:
    - 按前缀分组的数据字典
    """
    # 将数据和标签展平
    flat_data = data.flatten()
    flat_labels = labels.flatten()
    
    # 过滤掉空值标签
    valid_mask = (flat_labels != '') & (flat_labels != None)
    flat_data = flat_data[valid_mask]
    flat_labels = flat_labels[valid_mask]
    
    # 将标签转为字符串，只进行一次转换
    str_labels = np.array([str(label) for label in flat_labels])
    
    # 创建结果字典
    result_dict = {}
    
    for prefix in prefixes:
        # 一次性判断前缀和是否含有效索引
        prefix_digit_masks = [(label.startswith(prefix) and 
                            len(label) > len(prefix) and
                            label[len(prefix):].isdigit()) 
                            for label in str_labels]
        prefix_digit_mask = np.array(prefix_digit_masks)
        
        if not np.any(prefix_digit_mask):
            print(f"警告: 没有找到前缀{prefix}的有效数据!")
            result_dict[prefix] = np.array([])
            continue
        
        # 提取索引和对应数据
        matched_data = flat_data[prefix_digit_mask]
        
        # 直接从标签中提取索引
        indices = np.array([int(label[len(prefix):]) for label in str_labels[prefix_digit_mask]])
        
        # 创建结果数组并高效填充
        if indices.size > 0:
            max_idx = np.max(indices)
            result = np.zeros(max_idx + 1)
            # 使用numpy的高级索引功能高效累加
            np.add.at(result, indices, matched_data)
            result_dict[prefix] = result
        else:
            result_dict[prefix] = np.array([])
    
    return result_dict
