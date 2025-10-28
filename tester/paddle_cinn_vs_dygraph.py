import paddle
from paddle.jit import to_static

from .api_config.log_writer import write_to_log
from .base import APITestBase, CUDA_ERRORS


class APITestCINNVSDygraph(APITestBase):
    def __init__(self, api_config, **kwargs):
        super().__init__(api_config)
        self.test_amp = kwargs.get("test_amp", False)

    def test(self):

        if self.need_skip():
            print("[Skip]", flush=True)
            return

        if not self.ana_paddle_api_info():
            print("ana_paddle_api_info failed", flush=True)
            return

        try:
            if not self.gen_numpy_input():
                print("gen_numpy_input failed", flush=True)
                return
        except Exception as err:
            print(f"[numpy error] {self.api_config.config}\n{str(err)}", flush=True)
            write_to_log("numpy_error", self.api_config.config)
            return

        try:

            def func_backward(result_outputs, inputs_list, result_outputs_grads):
                return paddle.grad(
                    result_outputs,
                    inputs_list,
                    grad_outputs=result_outputs_grads,
                    allow_unused=True,
                )

            build_strategy = paddle.static.BuildStrategy()
            build_strategy.build_cinn_pass = True

            @to_static(full_graph=True, build_strategy=build_strategy)
            def func_backward_static(result_outputs, inputs_list, result_outputs_grads):
                return func_backward(result_outputs, inputs_list, result_outputs_grads)

            def func(args, kwargs):
                if self.api_config.api_name.startswith("paddle.Tensor."):
                    api_name = self.api_config.api_name.split(".")[-1]
                    api = getattr(args[0], api_name)
                    return api(*args[1:], **kwargs)
                return self.paddle_api(*args, **kwargs)

            @to_static(full_graph=True, build_strategy=build_strategy)
            def func_static(args, kwargs):
                return func(args, kwargs)

            if not self.gen_paddle_input():
                print("gen_paddle_input failed", flush=True)
                return

            if self.test_amp:
                with paddle.amp.auto_cast():
                    paddle_output = func(self.paddle_args, self.paddle_kwargs)
                    paddle_output_static = func_static(
                        self.paddle_args, self.paddle_kwargs
                    )
            else:
                paddle_output = func(self.paddle_args, self.paddle_kwargs)
                paddle_output_static = func_static(self.paddle_args, self.paddle_kwargs)
        except Exception as err:
            if self.should_ignore_paddle_error(str(err)):
                print(f"[Pass] {self.api_config.config}", flush=True)
                write_to_log("pass", self.api_config.config)
                return
            print(f"[paddle error] {self.api_config.config}\n{str(err)}", flush=True)
            write_to_log("paddle_error", self.api_config.config)
            if any(cuda_err in str(err) for cuda_err in CUDA_ERRORS):
                raise
            return

        try:
            paddle.base.core.eager._for_test_check_cuda_error()
        except Exception as err:
            print(f"[cuda error] {self.api_config.config}\n{str(err)}", flush=True)
            write_to_log("paddle_error", self.api_config.config)
            return

        if self.api_config.api_name == "paddle.broadcast_shape":
            return

        if not self.compare(paddle_output, paddle_output_static):
            return

        if self.need_check_grad():
            self.is_backward = True
            try:
                out_grads = None
                out_grads_static = None
                inputs_list = self.get_paddle_input_list()
                result_outputs, result_outputs_grads = (
                    self.gen_paddle_output_and_output_grad(paddle_output)
                )
                if inputs_list and result_outputs and result_outputs_grads:
                    out_grads = func_backward(
                        result_outputs, inputs_list, result_outputs_grads
                    )
                    out_grads_static = func_backward_static(
                        result_outputs, inputs_list, result_outputs_grads
                    )
            except Exception as err:
                if str(err).startswith("Too large tensor to get cached numpy: "):
                    print(
                        f"[numpy error] backward {self.api_config.config}\n{str(err)}",
                        flush=True,
                    )
                    write_to_log("numpy_error", self.api_config.config)
                    return
                if self.should_ignore_paddle_error(str(err)):
                    print(f"[Pass] {self.api_config.config}", flush=True)
                    write_to_log("pass", self.api_config.config)
                    return
                print(
                    f"[paddle error] backward {self.api_config.config}\n{str(err)}",
                    flush=True,
                )
                write_to_log("paddle_error", self.api_config.config)
                if any(cuda_err in str(err) for cuda_err in CUDA_ERRORS):
                    raise
                return

            try:
                paddle.base.core.eager._for_test_check_cuda_error()
            except Exception as err:
                print(
                    f"[cuda error] backward {self.api_config.config}\n{str(err)}",
                    flush=True,
                )
                write_to_log("paddle_error", self.api_config.config)
                raise

            if not self.compare(out_grads, out_grads_static, is_backward=True):
                return

        print("[Pass]", self.api_config.config, flush=True)
        write_to_log("pass", self.api_config.config)

    def compare(self, dygraph_output, static_output, is_backward=False):
        backward_str = "backward " if is_backward else ""
        if isinstance(dygraph_output, paddle.Tensor):
            if not isinstance(static_output, paddle.Tensor):
                print(
                    f"[accuracy error] {backward_str}{self.api_config.config}\n[not compare] type,",
                    f"dygraph: {type(dygraph_output)}, static: {type(static_output)}",
                    flush=True,
                )
                write_to_log("accuracy_error", self.api_config.config)
                return False
            try:
                self.paddle_assert_accuracy(dygraph_output, static_output)
            except Exception as err:
                print(
                    f"[accuracy error] {backward_str}{self.api_config.config}\n{str(err)}",
                    flush=True,
                )
                write_to_log("accuracy_error", self.api_config.config)
                return False
        elif isinstance(dygraph_output, (list, tuple)):
            if not isinstance(static_output, (list, tuple)):
                print(
                    f"[accuracy error] {backward_str}{self.api_config.config}\n[not compare] type,",
                    f"dygraph: {type(dygraph_output)}, static: {type(static_output)}",
                    flush=True,
                )
                write_to_log("accuracy_error", self.api_config.config)
                return False
            dygraph_output = list(dygraph_output)
            static_output = list(static_output)
            if len(dygraph_output) != len(static_output):
                print(
                    f"[accuracy error] {backward_str}{self.api_config.config}\n[not compare] length,",
                    f"dygraph: {len(dygraph_output)}, static: {len(static_output)}",
                    flush=True,
                )
                write_to_log("accuracy_error", self.api_config.config)
                return False
            for i in range(len(dygraph_output)):
                if not isinstance(dygraph_output[i], paddle.Tensor) or not isinstance(
                    static_output[i], paddle.Tensor
                ):
                    print(
                        f"[accuracy error] {backward_str}{self.api_config.config}\n[not compare] type at {i}-th,",
                        f"dygraph: {type(dygraph_output[i])}, static: {type(static_output[i])}",
                        flush=True,
                    )
                    write_to_log("accuracy_error", self.api_config.config)
                    return False
                try:
                    self.paddle_assert_accuracy(dygraph_output[i], static_output[i])
                except Exception as err:
                    print(
                        f"[accuracy error] {backward_str}{self.api_config.config}\n{str(err)}",
                        flush=True,
                    )
                    write_to_log("accuracy_error", self.api_config.config)
                    return False
        return True
