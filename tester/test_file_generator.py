"""
单测文件生成器模块

用于在测试失败时自动生成可复现的单测文件
"""
import os
import hashlib
import numpy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


def _serialize_numpy_array(arr: numpy.ndarray, var_name: str, max_size_for_inline: int = 1000) -> str:
    """
    将numpy数组序列化为Python代码
    
    参数:
        arr: numpy数组
        var_name: 变量名
        max_size_for_inline: 超过此大小的数组将使用numpy.load加载
    
    返回:
        Python代码字符串
    """
    numel = arr.size
    if numel > max_size_for_inline:
        # 对于大数组，使用numpy.save保存，然后加载
        # 这里我们直接生成数组数据，但使用更紧凑的格式
        return (f"# Large array: shape={arr.shape}, dtype={arr.dtype}\n"
                f"{var_name} = numpy.array({arr.tolist()}, dtype='{arr.dtype}').reshape({arr.shape})")
    else:
        # 对于小数组，直接内联
        if arr.size == 0:
            return f"{var_name} = numpy.array([], dtype='{arr.dtype}').reshape({arr.shape})"
        elif arr.size == 1:
            return f"{var_name} = numpy.array({arr.item()}, dtype='{arr.dtype}')"
        else:
            # 使用tolist()转换为Python列表
            list_repr = arr.tolist()
            return f"{var_name} = numpy.array({list_repr}, dtype='{arr.dtype}').reshape({arr.shape})"


def _extract_numpy_from_config_item(config_item):
    """从单个配置项中提取numpy数据"""
    from .api_config.config_analyzer import TensorConfig
    
    if isinstance(config_item, TensorConfig):
        if hasattr(config_item, 'numpy_tensor') and config_item.numpy_tensor is not None:
            return config_item.numpy_tensor
    elif isinstance(config_item, (list, tuple)):
        result = []
        for item in config_item:
            extracted = _extract_numpy_from_config_item(item)
            if extracted is not None:
                result.append(extracted)
        return tuple(result) if isinstance(config_item, tuple) else result
    return None


def _extract_numpy_inputs(api_config, args_config, kwargs_config) -> Tuple[List[Tuple[str, numpy.ndarray]], Dict[str, numpy.ndarray]]:
    """
    从API配置中提取所有numpy输入数据
    
    返回:
        (args_numpy, kwargs_numpy): 位置参数和关键字参数的numpy数据
    """
    from .api_config.config_analyzer import TensorConfig
    
    args_numpy = []
    kwargs_numpy = {}
    
    def extract_from_config(config_item, index=None, key=None):
        """递归提取numpy数据"""
        if isinstance(config_item, TensorConfig):
            # 检查是否是TensorConfig类型
            if hasattr(config_item, 'numpy_tensor') and config_item.numpy_tensor is not None:
                return config_item.numpy_tensor
        elif isinstance(config_item, (list, tuple)):
            # 处理列表或元组
            result = []
            for i, item in enumerate(config_item):
                extracted = extract_from_config(item, index=i)
                if extracted is not None:
                    result.append(extracted)
            return tuple(result) if isinstance(config_item, tuple) else result
        return None
    
    # 提取位置参数
    for i, arg_config in enumerate(args_config):
        numpy_data = extract_from_config(arg_config, index=i)
        if numpy_data is not None:
            args_numpy.append((f"arg_{i}", numpy_data))
    
    # 提取关键字参数
    for key, kwarg_config in kwargs_config.items():
        numpy_data = extract_from_config(kwarg_config, key=key)
        if numpy_data is not None:
            kwargs_numpy[key] = numpy_data
    
    return args_numpy, kwargs_numpy


def _generate_test_code(
    api_name: str,
    api_config_str: str,
    args_numpy: List[Tuple[str, numpy.ndarray]],
    kwargs_numpy: Dict[str, numpy.ndarray],
    error_info: Dict[str, Any],
    test_amp: bool = False,
    target_device: str = "xpu",
    device_id: int = 0,
    non_tensor_args: List[Tuple[int, Any]] = None,
    non_tensor_kwargs: Dict[str, Any] = None,
) -> str:
    """
    生成单测文件代码
    
    参数:
        api_name: API名称
        args_numpy: 位置参数的numpy数据列表
        kwargs_numpy: 关键字参数的numpy数据字典
        error_info: 错误信息字典
        test_amp: 是否测试AMP
        target_device: 目标设备类型
        device_id: 设备ID
    """
    # 生成文件头
    code_lines = [
        '"""',
        f'自动生成的单测文件 - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        f'API: {api_name}',
        f'配置: {api_config_str[:200]}...' if len(api_config_str) > 200 else f'配置: {api_config_str}',
        f'错误类型: {error_info.get("error_type", "unknown")}',
        f'失败阶段: {error_info.get("stage", "unknown")}',
        '"""',
        '',
        'import paddle',
        'import numpy',
        'import torch',
        '',
        f'# 设置目标设备',
        f'paddle.set_device("{target_device}:{device_id}")',
        '',
    ]
    
    # 检查是否是Tensor方法调用
    is_tensor_method = api_name.startswith('paddle.Tensor.')
    
    # 对于Tensor方法，如果args为空，self可能在kwargs中
    if is_tensor_method and not args_numpy and kwargs_numpy:
        # 将第一个kwargs项移到args中作为self
        first_key = next(iter(kwargs_numpy.keys()))
        first_value = kwargs_numpy.pop(first_key)
        args_numpy.insert(0, (first_key, first_value))
    
    # 生成输入数据
    code_lines.append('# 生成输入数据')
    all_inputs = []
    tensor_vars = {}  # 存储tensor变量名
    
    # 处理位置参数
    for i, (var_name, numpy_data) in enumerate(args_numpy):
        if isinstance(numpy_data, (list, tuple)):
            # 处理列表/元组类型的输入
            code_lines.append(f'# 位置参数 {var_name} (list/tuple)')
            tensor_list = []
            for j, item in enumerate(numpy_data):
                if isinstance(item, numpy.ndarray):
                    item_var = f'{var_name}_item_{j}'
                    code_lines.append(_serialize_numpy_array(item, item_var))
                    # 创建paddle tensor
                    tensor_var = f'{item_var}_tensor'
                    code_lines.append(f'{tensor_var} = paddle.to_tensor({item_var})')
                    tensor_list.append(tensor_var)
                    all_inputs.append(tensor_var)
            if tensor_list:
                code_lines.append(f'{var_name}_tensors = [{", ".join(tensor_list)}]')
        elif isinstance(numpy_data, numpy.ndarray):
            code_lines.append(f'# 位置参数 {var_name}')
            code_lines.append(_serialize_numpy_array(numpy_data, var_name))
            # 创建paddle tensor
            tensor_var = f'{var_name}_tensor'
            code_lines.append(f'{tensor_var} = paddle.to_tensor({var_name})')
            tensor_vars[var_name] = tensor_var
            all_inputs.append(tensor_var)
    
    # 处理关键字参数
    for key, numpy_data in kwargs_numpy.items():
        if isinstance(numpy_data, (list, tuple)):
            # 处理列表/元组类型的关键字参数
            code_lines.append(f'# 关键字参数 {key} (list/tuple)')
            tensor_list = []
            for j, item in enumerate(numpy_data):
                if isinstance(item, numpy.ndarray):
                    item_var = f'kwarg_{key}_item_{j}'
                    code_lines.append(_serialize_numpy_array(item, item_var))
                    # 创建paddle tensor
                    tensor_var = f'{item_var}_tensor'
                    code_lines.append(f'{tensor_var} = paddle.to_tensor({item_var})')
                    tensor_list.append(tensor_var)
                    all_inputs.append(tensor_var)
            if tensor_list:
                code_lines.append(f'kwarg_{key}_tensors = [{", ".join(tensor_list)}]')
        elif isinstance(numpy_data, numpy.ndarray):
            code_lines.append(f'# 关键字参数 {key}')
            var_name = f'kwarg_{key}'
            code_lines.append(_serialize_numpy_array(numpy_data, var_name))
            # 创建paddle tensor
            tensor_var = f'{var_name}_tensor'
            code_lines.append(f'{tensor_var} = paddle.to_tensor({var_name})')
            tensor_vars[key] = tensor_var
            all_inputs.append(tensor_var)
    
    code_lines.append('')
    code_lines.append('# 构建API调用参数')
    
    # 处理非tensor参数
    if non_tensor_args is None:
        non_tensor_args = []
    if non_tensor_kwargs is None:
        non_tensor_kwargs = {}
    
    # 为非tensor参数生成代码
    for idx, value in non_tensor_args:
        # 非tensor参数直接使用原值
        code_lines.append(f'# 非tensor位置参数 arg_{idx}')
        if isinstance(value, str):
            code_lines.append(f'arg_{idx}_non_tensor = "{value}"')
        else:
            code_lines.append(f'arg_{idx}_non_tensor = {repr(value)}')
    
    for key, value in non_tensor_kwargs.items():
        # 非tensor关键字参数直接使用原值
        code_lines.append(f'# 非tensor关键字参数 {key}')
        if isinstance(value, str):
            code_lines.append(f'kwarg_{key}_non_tensor = "{value}"')
        else:
            code_lines.append(f'kwarg_{key}_non_tensor = {repr(value)}')
    
    code_lines.append('')
    
    # 构建位置参数列表（使用tensor变量）
    arg_vars = []
    numpy_idx = 0
    non_tensor_idx = 0
    for i in range(max(len(args_numpy) + len(non_tensor_args), 0)):
        if numpy_idx < len(args_numpy) and args_numpy[numpy_idx][0] == f"arg_{i}":
            # 这是tensor参数
            var_name, numpy_data = args_numpy[numpy_idx]
            if isinstance(numpy_data, (list, tuple)):
                arg_vars.append(f'{var_name}_tensors')
            else:
                arg_vars.append(tensor_vars.get(var_name, var_name))
            numpy_idx += 1
        elif non_tensor_idx < len(non_tensor_args) and non_tensor_args[non_tensor_idx][0] == i:
            # 这是非tensor参数
            arg_vars.append(f'arg_{i}_non_tensor')
            non_tensor_idx += 1
    
    # 构建关键字参数字典（使用tensor变量）
    kwarg_vars = {}
    for key, numpy_data in kwargs_numpy.items():
        if isinstance(numpy_data, (list, tuple)):
            kwarg_vars[key] = f'kwarg_{key}_tensors'
        else:
            kwarg_vars[key] = tensor_vars.get(key, f'kwarg_{key}_tensor')
    
    # 添加非tensor关键字参数
    for key in non_tensor_kwargs:
        kwarg_vars[key] = f'kwarg_{key}_non_tensor'
    
    # 生成API调用代码
    code_lines.append('')
    code_lines.append('# 执行API调用')
    code_lines.append('try:')
    
    # 检查是否是Tensor方法调用
    is_tensor_method = api_name.startswith('paddle.Tensor.')
    if is_tensor_method:
        # 对于Tensor方法，第一个参数是self（tensor对象）
        method_name = api_name.split('.')[-1]
        if arg_vars:
            tensor_var = arg_vars[0]
            remaining_args = arg_vars[1:] if len(arg_vars) > 1 else []
        else:
            # 如果没有位置参数，尝试从kwargs中获取（通常是self）
            # 对于Tensor方法，self通常在kwargs中
            tensor_var = None
            remaining_args = []
    else:
        tensor_var = None
        remaining_args = arg_vars
    
    # 构建API调用
    api_call_parts = []
    if not is_tensor_method:
        # 普通API调用
        if remaining_args:
            api_call_parts.extend(remaining_args)
    else:
        # Tensor方法调用
        if remaining_args:
            api_call_parts.extend(remaining_args)
    
    if kwarg_vars:
        kwarg_str = ', '.join([f'{k}={v}' for k, v in kwarg_vars.items()])
        api_call_parts.append(kwarg_str)
    
    if is_tensor_method and tensor_var:
        # Tensor方法调用: tensor.method(*args, **kwargs)
        if api_call_parts:
            api_call = f'    output = {tensor_var}.{method_name}(' + ', '.join(api_call_parts) + ')'
        else:
            api_call = f'    output = {tensor_var}.{method_name}()'
    else:
        # 普通API调用: api_name(*args, **kwargs)
        if api_call_parts:
            api_call = f'    output = {api_name}(' + ', '.join(api_call_parts) + ')'
        else:
            api_call = f'    output = {api_name}()'
    
    if test_amp:
        code_lines.append('    with paddle.amp.auto_cast():')
        code_lines.append('        ' + api_call)
    else:
        code_lines.append(api_call)
    
    code_lines.append('    print("Forward pass succeeded")')
    code_lines.append('    print(f"Output type: {type(output)}")')
    code_lines.append('    if isinstance(output, paddle.Tensor):')
    code_lines.append('        print(f"Output shape: {output.shape}, dtype: {output.dtype}")')
    code_lines.append('    elif isinstance(output, (list, tuple)):')
    code_lines.append('        print(f"Output length: {len(output)}")')
    code_lines.append('        for i, item in enumerate(output):')
    code_lines.append('            if isinstance(item, paddle.Tensor):')
    code_lines.append('                print(f"  Output[{i}]: shape={item.shape}, dtype={item.dtype}")')
    code_lines.append('')
    
    # 如果有backward测试
    if error_info.get("stage") == "backward" or error_info.get("need_backward", False):
        code_lines.append('')
        code_lines.append('    # Backward测试')
        code_lines.append('    if isinstance(output, paddle.Tensor):')
        code_lines.append('        output.backward()')
        code_lines.append('    elif isinstance(output, (list, tuple)):')
        code_lines.append('        for item in output:')
        code_lines.append('            if isinstance(item, paddle.Tensor):')
        code_lines.append('                item.backward()')
        code_lines.append('    print("Backward pass succeeded")')
    
    code_lines.append('except Exception as e:')
    code_lines.append('    print(f"Error occurred: {e}")')
    code_lines.append('    import traceback')
    code_lines.append('    traceback.print_exc()')
    code_lines.append('    raise')
    
    return '\n'.join(code_lines)


def generate_reproducible_test_file(
    api_config,
    error_info: Dict[str, Any],
    output_dir: str = "failed_tests",
    test_amp: bool = False,
    target_device: str = "xpu",
    device_id: int = 0,
    test_instance=None,
) -> Optional[str]:
    """
    生成可复现的单测文件
    
    参数:
        api_config: APIConfig对象
        error_info: 错误信息字典，包含error_type, stage等
        output_dir: 输出目录
        test_amp: 是否测试AMP
        target_device: 目标设备类型
        device_id: 设备ID
        test_instance: 测试类实例，用于提取numpy数据
    
    返回:
        生成的测试文件路径，如果失败则返回None
    """
    try:
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 优先从测试实例中提取numpy数据（因为此时数据已经生成）
        args_numpy = []
        kwargs_numpy = {}
        non_tensor_args = []
        non_tensor_kwargs = {}
        
        if test_instance is not None:
            # 从测试实例的paddle_args_config和paddle_kwargs_config中提取
            if hasattr(test_instance, 'paddle_args_config'):
                for i, arg_config in enumerate(test_instance.paddle_args_config):
                    numpy_data = _extract_numpy_from_config_item(arg_config)
                    if numpy_data is not None:
                        args_numpy.append((f"arg_{i}", numpy_data))
                    else:
                        # 非tensor参数
                        from .api_config.config_analyzer import TensorConfig
                        if not isinstance(arg_config, TensorConfig):
                            non_tensor_args.append((i, arg_config))
            
            if hasattr(test_instance, 'paddle_kwargs_config'):
                for key, kwarg_config in test_instance.paddle_kwargs_config.items():
                    numpy_data = _extract_numpy_from_config_item(kwarg_config)
                    if numpy_data is not None:
                        kwargs_numpy[key] = numpy_data
                    else:
                        # 非tensor参数
                        from .api_config.config_analyzer import TensorConfig
                        if not isinstance(kwarg_config, TensorConfig):
                            non_tensor_kwargs[key] = kwarg_config
        
        # 如果没有从测试实例中提取到数据，尝试从api_config中提取
        if not args_numpy and not kwargs_numpy:
            args_numpy, kwargs_numpy = _extract_numpy_inputs(
                api_config,
                api_config.args if hasattr(api_config, 'args') else [],
                api_config.kwargs if hasattr(api_config, 'kwargs') else {}
            )
        
        # 生成文件名
        api_name_safe = api_config.api_name.replace('.', '_').replace(':', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_hash = hashlib.md5(api_config.config.encode()).hexdigest()[:8]
        pid = os.getpid()
        filename = f"test_{api_name_safe}_{timestamp}_{pid}_{config_hash}.py"
        filepath = output_path / filename
        
        # 生成测试代码
        test_code = _generate_test_code(
            api_config.api_name,
            api_config.config,
            args_numpy,
            kwargs_numpy,
            error_info,
            test_amp=test_amp,
            target_device=target_device,
            device_id=device_id,
            non_tensor_args=non_tensor_args,
            non_tensor_kwargs=non_tensor_kwargs,
        )
        
        # 写入文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(test_code)
        
        print(f"[Generated test file] {filepath}", flush=True)
        return str(filepath)
        
    except Exception as e:
        print(f"[Error generating test file] {e}", flush=True)
        import traceback
        traceback.print_exc()
        return None

