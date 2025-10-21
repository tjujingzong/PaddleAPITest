# API 调用追踪器 (API Call Tracer)

这是一个用于在运行时动态捕获框架 API 调用的工具。现已实现 PyTorch 原生底层的追踪功能。

其主要功能是抓取 API 的实际调用参数（配置），并生成配置集，可以为后续的模糊测试、随机测试或 API 重放提供数据支持。

## 主要功能

1. 能够追踪 `torch` 的原生函数和底层算子（即所有通过 `__torch_function__` 协议暴露的算子），而非 `python` 封装函数
2. 通过挂钩（Hook）机制，在代码实际执行时动态捕获每一次 API 调用及其参数
3. 将捕获到的 API 调用信息序列化为两种格式并保存：
    - `api_trace.yaml`: 结构化数据，便于序列化与反序列化
    - `api_trace.txt`: 人类可读格式，便于测试去重

## 技术设计与实现

### 设计原则

- **High Decoupling**:

    **`api_tracer.py`**:
    - `APITracer`: 顶层控制器，负责追踪任务的生命周期管理（启动、停止）

    **`framework_dialect.py`**:
    - `TracingHook`: 钩子实现类，定义了具体的API拦截策略
    - `FrameworkDialect`: 框架方言组，封装特定框架（如PyTorch）的独有逻辑，如特殊类型序列化

    **`config_serializer.py`**:
    - `ConfigSerializer`: 序列化类，负责将捕获的数据格式化并写入文件

- **High Extensibility**:

    - `TracingHook` : 允许开发者通过继承该基类，来实现全新的API挂钩策略
    - `FrameworkDialect` : 用户可以继承该基类，通过重写方法来快速支持新的深度学习框架

### 多功能钩子策略

- `SetattrHook`:
    通过 Python 的 `setattr` 机制，在运行时动态遍历并替换模块中的函数对象。它可以挂钩任意 Python 库的 API。可用于扫描全库并产出 `api_list/torch_api_list_full.yaml` ，追踪纯 Python 函数的调用。
    
    由于 PyTorch 的大量核心功能由 C++ 实现，`SetattrHook` 钩子无法覆盖到底层算子，但可以抓取如 `nn.Linear` 等类级别 API。覆盖范围默认是 `api_list/torch_api_list.yaml` 的子集，由参数 `disable_torch_api_list` 控制。

- `TorchFunctionHook`:
    通过重写 PyTorch 官方的 `torch.overrides.TorchFunctionMode` 类实现。该方法可以捕获所有支持 `__torch_function__` 协议的 API 调用，即进入 PyTorch C++ 后端的函数调用。经过测试，这是追踪 PyTorch API 的首选方法（目前采用 `SetattrHook` + `TorchFunctionHook` 结合的方式），能够高效、准确地捕获所有 Torch C API 调用，覆盖范围广、对用户代码无侵入。

- `TorchDispatchHook`:
    通过重写 PyTorch 官方的 `torch.utils._python_dispatch` 库的 `TorchDispatchMode` 类实现。该方法可以捕获所有通过 `torch.dispatch` 调用的函数，包括自定义的 `Tensor` 操作。 `torch.dispatch` 是 PyTorch 内部使用的调度机制，可以捕获到所有底层算子的调用（如 `aten::`），是抓取 PyTorch 底层算子的首选方法。

### PyTorch 其他定制实现

- `PyTorchDialect` 实现了 `FrameworkDialect` 抽象类 ，其方法 `serialize_special_type` 、 `format_special_type` 实现了针对 `torch.Tensor` 、 `torch.dtype` 、 `torch.device` 等 PyTorch 特有类型的序列化逻辑

## 如何使用

使用 `APITracer` 非常简单，只需将其作为一个上下文管理器（Context Manager）包裹住需要追踪的 PyTorch 代码即可。

**示例代码:**

```python
import torch
from api_tracer import APITracer

# 初始化 Tracer，指定框架方言为 'torch'
tracer = APITracer(
    dialect="torch", output_path="trace_output", levels=1, merge_output=True
)

# 使用 with 语句来自动启动和停止追踪
with tracer:
    # 执行 PyTorch 模型或代码
    tensor1 = torch.randn(2, 3, device="cpu")
    tensor2 = torch.ones(2, 3)
    result = torch.add(tensor1, tensor2, alpha=10)
    final_sum = result.sum()

# 或者手动启动和停止追踪：
# tracer.start()
# 执行 PyTorch 模型或代码
# tracer.stop()

```

**参数说明**

- `dialect` (str): 支持的框架方言，目前仅支持 `torch`
- `output_path` (str): 抓取结果的保存目录路径
- `levels` (int|List[int]): 控制钩子的粒度，可同时启用多个钩子，默认为 `0` 。映射如下:
  - `0`: `SetattrHook`
  - `1`: `TorchFunctionHook`
  - `2`: `TorchDispatchHook`

**可选参数**
- `merge_output` (bool): 输出时是否将不同 level 的结果合并，默认为 `False`
- `record_stack` (bool): 是否记录调用栈信息，默认为 `False`
- `stack_format` (str): 指定调用栈信息的格式， `full` 为 traceback 样式， `short` 为简化的 traceback， `api` 为模块式的 API 样式。默认为 `short`
- `disable_torch_api_list` (bool): 是否禁用 `torch_api_list` ，仅影响 `PyTorchDialect` 的 `SetattrHook` 钩子。设置为 `True` 时将抓取所有遍历到并被 `setattr` 钩住的 API ，除非在 `PyTorchDialect` 中被排除。默认为 `False`

> [!CAUTION]
> 目前已知 `SetattrHook` 与 `torch.compile` 或 `@functools.wraps` 等复杂场景交互时，部分 `staticmethod` 方法会产生绑定错误。例如：
> ```bash
> TypeError: Node._pretty_print_target() takes 1 positional argument but 2 were given
> ```
> 最佳的处理方式是将相关 API 添加至 `framework_dialect.py / IGNORE_CLASSES_OR_METHODS` 列表中，单纯跳过；若修改 `_create_wrapper` 方法，采用 `inspect.signature` 可能会增加绑定负担，也可能会有更多问题 ：）
>
> 此外，`TorchFunctionHook` 钩子已默认跳过 `torch.overrides.get_ignored_functions()` 列表的函数，这意味着大多数工厂函数 (如 `torch.randn` ) 会被跳过以避免无法预知的错误 (尽管注释后通常不会引起错误)，跳过的 API 可以由 `SetattrHook` 捕获

### 输出文件

执行上述代码后，你将在 `trace_output` 目录下找到五个文件（若开启 `record_stack` 则有六个）：

1. **`api_trace.yaml`**: 结构化的 API 调用记录

    ```yaml
    - api: torch.randn
      args:
      - 2
      - 3
      kwargs:
        device: cpu
    - api: torch.ones
      args:
      - 2
      - 3
      kwargs: {}
    - api: torch.add
      args:
      - type: torch.Tensor
        shape:
        - 2
        - 3
        dtype: torch.float32
        device: cpu
      - type: torch.Tensor
        shape:
        - 2
        - 3
        dtype: torch.float32
        device: cpu
      kwargs:
        alpha: 10
    - api: torch.Tensor.sum
      args:
      - type: torch.Tensor
        shape:
        - 2
        - 3
        dtype: torch.float32
        device: cpu
      kwargs: {}
    ```

2. **`api_trace.txt`**: 更易读的格式

    ```text
    torch.randn(2, 3, device="cpu")
    torch.ones(2, 3)
    torch.add(Tensor([2, 3], "float32"), Tensor([2, 3], "float32"), alpha=10)
    torch.Tensor.sum(Tensor([2, 3], "float32"))
    ```
3. **`api_apis.txt`**: API 集合

    ```text
    torch.Tensor.sum
    torch.add
    torch.ones
    torch.randn
    ```

4. **`api_configs.yaml`**: API 配置集合（去重排序 `api_trace.txt` ）

    ```text
    torch.Tensor.sum(Tensor([2, 3], "float32"))
    torch.add(Tensor([2, 3], "float32"), Tensor([2, 3], "float32"), alpha=10)
    torch.ones(2, 3)
    torch.randn(2, 3, device="cpu")
    ```

5. **`api_statistics.yaml`**: API 统计信息

    ```text
    Total APIs: 4
    Total API calls: 4

    torch.randn: 1 (25.00%)
    torch.ones: 1 (25.00%)
    torch.add: 1 (25.00%)
    torch.Tensor.sum: 1 (25.00%)
    ```

6. **`api_stacks.yaml`** (仅开启 `record_stack` ): API 调用栈信息

    ```yaml
    torch.Tensor.sum:
      3stacks:
      - - test.py:19 in <module>
    torch.add:
      3stacks:
      - - test.py:18 in <module>
    torch.ones:
      3stacks:
      - - test.py:17 in <module>
    torch.randn:
      3stacks:
      - - test.py:16 in <module>
      ```
