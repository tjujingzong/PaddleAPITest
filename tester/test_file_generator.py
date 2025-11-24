"""
单测文件生成器模块

用于在测试失败时自动生成可复现的单测文件
"""
import os
import re
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
    output_grads_numpy: Optional[List[numpy.ndarray]] = None,
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
    is_tensor_method = api_name.startswith('paddle.Tensor.')
    method_name = api_name.split(".")[-1] if is_tensor_method else ""
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
        f'TARGET_DEVICE = "{target_device}:{device_id}"',
        'RTOL = 1e-2',
        'ATOL = 1e-2',
        f'TEST_AMP = {str(test_amp)}',
        f'IS_TENSOR_METHOD = {str(is_tensor_method)}',
        f'METHOD_NAME = "{method_name}"',
        f'API_NAME = "{api_name}"',
        '',
        'NUMPY_DATA = {}',
        'if not IS_TENSOR_METHOD:',
        '    API_FUNC = eval(API_NAME)',
        'else:',
        '    API_FUNC = None',
        '',
    ]
    
    # 对于Tensor方法，如果args为空，self可能在kwargs中
    if is_tensor_method and not args_numpy and kwargs_numpy:
        # 将第一个kwargs项移到args中作为self
        first_key = next(iter(kwargs_numpy.keys()))
        first_value = kwargs_numpy.pop(first_key)
        args_numpy.insert(0, (first_key, first_value))
    
    code_lines.append('# 生成输入数据 (仅保存 numpy 数据)')
    backward_required = error_info.get("stage") == "backward" or error_info.get("need_backward", False)

    fallback_negative_index = -1
    arg_spec_map: Dict[int, Dict[str, Any]] = {}
    kwarg_spec_map: Dict[str, Dict[str, Any]] = {}

    def get_arg_index(label: str) -> int:
        nonlocal fallback_negative_index
        match = re.match(r"arg_(\d+)", label)
        if match:
            return int(match.group(1))
        idx = fallback_negative_index
        fallback_negative_index -= 1
        return idx

    for var_name, numpy_data in args_numpy:
        arg_index = get_arg_index(var_name)
        if isinstance(numpy_data, (list, tuple)):
            item_specs = []
            for j, item in enumerate(numpy_data):
                if isinstance(item, numpy.ndarray):
                    item_var = f'{var_name}_item_{j}'
                    code_lines.append(_serialize_numpy_array(item, item_var))
                    code_lines.append(f"NUMPY_DATA['{item_var}'] = {item_var}")
                    item_specs.append({"type": "tensor", "data_key": item_var})
            arg_spec_map[arg_index] = {
                "type": "list",
                "items": item_specs,
                "is_tuple": isinstance(numpy_data, tuple),
            }
        elif isinstance(numpy_data, numpy.ndarray):
            code_lines.append(_serialize_numpy_array(numpy_data, var_name))
            code_lines.append(f"NUMPY_DATA['{var_name}'] = {var_name}")
            arg_spec_map[arg_index] = {"type": "tensor", "data_key": var_name}

    for key, numpy_data in kwargs_numpy.items():
        if isinstance(numpy_data, (list, tuple)):
            item_specs = []
            for j, item in enumerate(numpy_data):
                if isinstance(item, numpy.ndarray):
                    item_var = f'kwarg_{key}_item_{j}'
                    code_lines.append(_serialize_numpy_array(item, item_var))
                    code_lines.append(f"NUMPY_DATA['{item_var}'] = {item_var}")
                    item_specs.append({"type": "tensor", "data_key": item_var})
            kwarg_spec_map[key] = {
                "type": "list",
                "items": item_specs,
                "is_tuple": isinstance(numpy_data, tuple),
            }
        elif isinstance(numpy_data, numpy.ndarray):
            var_name = f'kwarg_{key}'
            code_lines.append(_serialize_numpy_array(numpy_data, var_name))
            code_lines.append(f"NUMPY_DATA['{var_name}'] = {var_name}")
            kwarg_spec_map[key] = {"type": "tensor", "data_key": var_name}
    
    if non_tensor_args is None:
        non_tensor_args = []
    if non_tensor_kwargs is None:
        non_tensor_kwargs = {}

    for idx, value in non_tensor_args:
        if idx not in arg_spec_map:
            arg_spec_map[idx] = {"type": "value", "value": value}
    for key, value in non_tensor_kwargs.items():
        if key not in kwarg_spec_map:
            kwarg_spec_map[key] = {"type": "value", "value": value}

    arg_specs = [arg_spec_map[idx] for idx in sorted(arg_spec_map.keys())]
    kwarg_specs = {key: kwarg_spec_map[key] for key in sorted(kwarg_spec_map.keys())}

    grad_keys: List[str] = []
    if output_grads_numpy:
        for idx, grad_numpy in enumerate(output_grads_numpy):
            var_name = f'output_grad_{idx}'
            code_lines.append(_serialize_numpy_array(grad_numpy, var_name))
            code_lines.append(f"NUMPY_DATA['{var_name}'] = {var_name}")
            grad_keys.append(var_name)

    code_lines.append('')
    code_lines.append(f'BACKWARD_REQUIRED = {str(backward_required)}')
    code_lines.append(f'ARG_SPECS = {repr(arg_specs)}')
    code_lines.append(f'KWARG_SPECS = {repr(kwarg_specs)}')
    code_lines.append(f'OUTPUT_GRAD_KEYS = {repr(grad_keys)}')
    code_lines.append('')

    helper_functions = [
        'def clone_tensor_from_data(key):',
        '    tensor = paddle.to_tensor(NUMPY_DATA[key])',
        '    tensor.stop_gradient = False',
        '    return tensor',
        '',
        'def build_from_spec(spec, tensor_refs):',
        '    spec_type = spec["type"]',
        '    if spec_type == "tensor":',
        '        tensor = clone_tensor_from_data(spec["data_key"])',
        '        tensor_refs.append(tensor)',
        '        return tensor',
        '    if spec_type == "list":',
        '        items = [build_from_spec(item, tensor_refs) for item in spec.get("items", [])]',
        '        return tuple(items) if spec.get("is_tuple") else items',
        '    if spec_type == "value":',
        '        return spec["value"]',
        '    raise ValueError(f"Unsupported spec type: {spec_type}")',
        '',
        'def build_inputs():',
        '    tensor_refs = []',
        '    args = [build_from_spec(spec, tensor_refs) for spec in ARG_SPECS]',
        '    kwargs = {key: build_from_spec(spec, tensor_refs) for key, spec in KWARG_SPECS.items()}',
        '    return args, kwargs, tensor_refs',
        '',
        'def collect_output_tensors(output):',
        '    tensors = []',
        '    if isinstance(output, paddle.Tensor):',
        '        if output._is_initialized() or output.numel() == 0:',
        '            tensors.append(output)',
        '        return tensors',
        '    if isinstance(output, (list, tuple)):',
        '        for item in output:',
        '            tensors.extend(collect_output_tensors(item))',
        '        return tensors',
        '    return tensors',
        '',
        'def build_output_grad_tensors(output):',
        '    tensors = collect_output_tensors(output)',
        '    if not tensors:',
        '        raise RuntimeError("Backward expected tensor outputs but none found")',
        '    if OUTPUT_GRAD_KEYS:',
        '        if len(OUTPUT_GRAD_KEYS) != len(tensors):',
        '            raise RuntimeError("Gradient spec count mismatch with outputs")',
        '        grad_tensors = []',
        '        for key, tensor in zip(OUTPUT_GRAD_KEYS, tensors):',
        '            grad_np = NUMPY_DATA[key]',
        '            grad_tensor = paddle.to_tensor(',
        '                grad_np,',
        '                dtype="float32" if tensor.dtype == paddle.bfloat16 else tensor.dtype,',
        '            )',
        '            if tensor.dtype == paddle.bfloat16:',
        '                grad_tensor = paddle.cast(grad_tensor, "bfloat16")',
        '            grad_tensors.append(grad_tensor)',
        '    else:',
        '        grad_tensors = [paddle.ones_like(tensor) for tensor in tensors]',
        '    return tensors, grad_tensors',
        '',
        'def run_backward(output):',
        '    tensors, grad_tensors = build_output_grad_tensors(output)',
        '    for tensor, grad in zip(tensors, grad_tensors):',
        '        tensor.backward(grad)',
        '    return',
        '',
        'def tensor_to_numpy(tensor):',
        '    if tensor.dtype == paddle.bfloat16:',
        '        tensor = paddle.cast(tensor, "float32")',
        '    return tensor.detach().cpu().numpy()',
        '',
        'def compare_tensors(cpu_tensor, target_tensor, label):',
        '    cpu_np = tensor_to_numpy(cpu_tensor)',
        '    target_np = tensor_to_numpy(target_tensor)',
        '    cpu_torch = torch.from_numpy(cpu_np)',
        '    target_torch = torch.from_numpy(target_np)',
        '    torch.testing.assert_close(',
        '        cpu_torch,',
        '        target_torch,',
        '        rtol=RTOL,',
        '        atol=ATOL,',
        '        equal_nan=True,',
        '        msg=f"{label} mismatch",',
        '    )',
        '',
        'def compare_outputs(cpu_output, target_output, prefix="output"):',
        '    if isinstance(cpu_output, paddle.Tensor):',
        '        if not isinstance(target_output, paddle.Tensor):',
        '            raise AssertionError(f"{prefix} type mismatch: {type(cpu_output)} vs {type(target_output)}")',
        '        compare_tensors(cpu_output, target_output, prefix)',
        '        return',
        '    if isinstance(cpu_output, (list, tuple)):',
        '        if not isinstance(target_output, (list, tuple)) or len(cpu_output) != len(target_output):',
        '            raise AssertionError(f"{prefix} length/type mismatch")',
        '        for idx, (cpu_item, target_item) in enumerate(zip(cpu_output, target_output)):',
        '            compare_outputs(cpu_item, target_item, f"{prefix}[{idx}]")',
        '        return',
        '    if cpu_output != target_output:',
        '        raise AssertionError(f"{prefix} mismatch: {cpu_output} vs {target_output}")',
        '',
        'def compare_gradients(cpu_grads, target_grads):',
        '    if len(cpu_grads) != len(target_grads):',
        '        raise AssertionError("Gradient list length mismatch")',
        '    for idx, (cpu_grad, target_grad) in enumerate(zip(cpu_grads, target_grads)):',
        '        if cpu_grad is None and target_grad is None:',
        '            continue',
        '        if (cpu_grad is None) != (target_grad is None):',
        '            raise AssertionError(f"Gradient[{idx}] presence mismatch")',
        '        cpu_torch = torch.from_numpy(cpu_grad)',
        '        target_torch = torch.from_numpy(target_grad)',
        '        torch.testing.assert_close(',
        '            cpu_torch,',
        '            target_torch,',
        '            rtol=RTOL,',
        '            atol=ATOL,',
        '            equal_nan=True,',
        '            msg=f"gradient[{idx}] mismatch",',
        '        )',
        '',
        'def run_on_device(device_str):',
        '    paddle.set_device(device_str)',
        '    args, kwargs, tensor_refs = build_inputs()',
        '    if IS_TENSOR_METHOD:',
        '        tensor_obj = args[0]',
        '        call_args = args[1:] if len(args) > 1 else []',
        '        api_callable = getattr(tensor_obj, METHOD_NAME)',
        '    else:',
        '        tensor_obj = None',
        '        call_args = args',
        '        api_callable = API_FUNC',
        '    if TEST_AMP:',
        '        with paddle.amp.auto_cast():',
        '            output = api_callable(*call_args, **kwargs)',
        '    else:',
        '        output = api_callable(*call_args, **kwargs)',
        '    grads = []',
        '    if BACKWARD_REQUIRED:',
        '        run_backward(output)',
        '        for tensor in tensor_refs:',
        '            grad = tensor.grad',
        '            grads.append(None if grad is None else grad.detach().cpu().numpy())',
        '    return output, grads',
    ]

    code_lines.extend(helper_functions)

    code_lines.append('cpu_output, cpu_grads = run_on_device("cpu")')
    code_lines.append('target_output, target_grads = run_on_device(TARGET_DEVICE)')
    code_lines.append('compare_outputs(cpu_output, target_output)')
    code_lines.append('print("Forward comparison passed")')
    if backward_required:
        code_lines.append('compare_gradients(cpu_grads, target_grads)')
        code_lines.append('print("Backward comparison passed")')
    code_lines.append('print("Reproduction completed.")')

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
        output_grads_numpy: List[numpy.ndarray] = []
        
        if test_instance is not None:
            if hasattr(test_instance, 'outputs_grad_numpy') and test_instance.outputs_grad_numpy:
                for grad_numpy in test_instance.outputs_grad_numpy:
                    if isinstance(grad_numpy, numpy.ndarray):
                        output_grads_numpy.append(grad_numpy)
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
        
        # 生成文件名（包含参数信息）
        api_name_safe = api_config.api_name.replace('.', '_').replace(':', '_')
        config_repr = re.sub(r'[^0-9a-zA-Z]+', '_', api_config.config)
        config_repr = re.sub(r'_+', '_', config_repr).strip('_')
        if len(config_repr) > 120:
            config_repr = config_repr[:120]
        if not config_repr:
            config_repr = api_name_safe
        config_hash = hashlib.md5(api_config.config.encode()).hexdigest()[:6]
        filename = f"test_{config_repr}_{config_hash}.py"
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
            output_grads_numpy=output_grads_numpy if output_grads_numpy else None,
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

