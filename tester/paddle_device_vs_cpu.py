import paddle
from .api_config.log_writer import write_to_log
from .base import APITestBase


class APITestCustomDeviceVSCPU(APITestBase):
    def __init__(self, api_config, **kwargs):
        super().__init__(api_config)
        self.test_amp = kwargs.get("test_amp", False)
        self.custom_device_type = self._get_first_custom_device_type()
        self.custom_device_id = 0
        
    def _get_first_custom_device_type(self):
        try:
            custom_device_types = paddle.device.get_all_custom_device_type()
            if custom_device_types:
                return custom_device_types[0]
            return "iluvatar_gpu"
        except Exception:
            return "iluvatar_gpu"
    
    def check_custom_device_available(self):
        """Check if CustomDevice is available"""
        try:
            custom_device_types = paddle.device.get_all_custom_device_type()
            return self.custom_device_type in custom_device_types
        except Exception:
            return False
    
    def run_on_device(self, device_type, device_id=0):
        """Run API on specified device"""
        try:
            if device_type == "cpu":
                paddle.set_device("cpu")
            else:
                paddle.set_device(f"{device_type}:{device_id}")

            if not self.gen_paddle_input():
                print(f"gen_paddle_input failed on {device_type}", flush=True)
                return None, None
            
            # Forward
            if self.test_amp:
                with paddle.amp.auto_cast():
                    output = self.paddle_api(*tuple(self.paddle_args), **self.paddle_kwargs)
            else:
                output = self.paddle_api(*tuple(self.paddle_args), **self.paddle_kwargs)
            
            # Backward
            out_grads = None
            if self.need_check_grad():
                inputs_list = self.get_paddle_input_list()
                result_outputs, result_outputs_grads = self.gen_paddle_output_and_output_grad(output)
                
                if inputs_list and result_outputs and result_outputs_grads:
                    try:
                        out_grads = paddle.grad(result_outputs, inputs_list, grad_outputs=result_outputs_grads, allow_unused=True)
                    except Exception as grad_err:
                        print(f"[{device_type} backward error]", self.api_config.config, "\n", str(grad_err), flush=True)
                        out_grads = None
                else:
                    print(f"[backward skip] No valid inputs or outputs for gradient computation on {device_type}", flush=True)
            
            return output, out_grads
            
        except Exception as err:
            print(f"[{device_type} error]", self.api_config.config, "\n", str(err), flush=True)
            return None, None
    
    def compare_outputs(self, cpu_output, custom_output):
        """Compare output results between CPU and CustomDevice"""
        if cpu_output is None or custom_output is None:
            print("[output none error]", self.api_config.config, flush=True)
            return False
            
        if isinstance(cpu_output, paddle.Tensor):
            if not isinstance(custom_output, paddle.Tensor):
                print("[output type diff error]", self.api_config.config, flush=True)
                return False
                
            try:
                # bfloat16 data type
                if cpu_output.dtype == paddle.bfloat16:
                    cpu_output = paddle.cast(cpu_output, dtype="float32")
                if custom_output.dtype == paddle.bfloat16:
                    custom_output = paddle.cast(custom_output, dtype="float32")
                
                custom_output_cpu = custom_output.cpu()

                self.np_assert_accuracy(cpu_output.numpy(), custom_output_cpu.numpy(), 1e-2, 1e-2)
                return True
                
            except Exception as err:
                print("[accuracy error]", self.api_config.config, "\n", str(err), flush=True)
                return False
                
        # list/tuple case
        elif isinstance(cpu_output, (list, tuple)):
            if not isinstance(custom_output, (list, tuple)):
                print("[output type diff error]", self.api_config.config, flush=True)
                return False
                
            # Convert to list
            if isinstance(cpu_output, tuple):
                cpu_output = list(cpu_output)
            if isinstance(custom_output, tuple):
                custom_output = list(custom_output)
                
            if len(cpu_output) != len(custom_output):
                print("[output length diff error]", self.api_config.config, flush=True)
                return False
                
            # Compare
            for i in range(len(cpu_output)):
                if not isinstance(cpu_output[i], paddle.Tensor):
                    print(f"skip non-tensor output[{i}]:", cpu_output[i], custom_output[i], flush=True)
                    continue
                    
                try:
                    # bfloat16
                    if cpu_output[i].dtype == paddle.bfloat16:
                        cpu_output[i] = paddle.cast(cpu_output[i], dtype="float32")
                    if custom_output[i].dtype == paddle.bfloat16:
                        custom_output[i] = paddle.cast(custom_output[i], dtype="float32")
                    
                    # Convert CustomDevice output to CPU
                    custom_output_cpu = custom_output[i].cpu()
                    
                    self.np_assert_accuracy(cpu_output[i].numpy(), custom_output_cpu.numpy(), 1e-2, 1e-2)
                    
                except Exception as err:
                    print("[accuracy error]", self.api_config.config, f" output[{i}]", "\n", str(err), flush=True)
                    return False
            
            return True
            
        else:
            # Non-Tensor output, print comparison directly
            print("non-tensor output comparison:", cpu_output, custom_output, flush=True)
            return True
    
    def compare_gradients(self, cpu_grads, custom_grads):
        """Compare gradient results between CPU and CustomDevice"""
        if cpu_grads is None or custom_grads is None:
            print("[gradients none error]", self.api_config.config, flush=True)
            return False
        
        # Convert to list for unified processing
        if isinstance(cpu_grads, paddle.Tensor):
            cpu_grads = [cpu_grads]
        if isinstance(custom_grads, paddle.Tensor):
            custom_grads = [custom_grads]
        
        if not isinstance(cpu_grads, (list, tuple)) or not isinstance(custom_grads, (list, tuple)):
            print("[gradients type error]", self.api_config.config, flush=True)
            return False
        
        # Convert to list
        cpu_grads = list(cpu_grads)
        custom_grads = list(custom_grads)
        
        if len(cpu_grads) != len(custom_grads):
            print("[gradients length diff error]", self.api_config.config, flush=True)
            return False
        
        # Compare gradients one by one
        for i, (cpu_grad, custom_grad) in enumerate(zip(cpu_grads, custom_grads)):
            if cpu_grad is None and custom_grad is None:
                continue
            elif cpu_grad is None or custom_grad is None:
                print(f"[gradient {i} none error]", self.api_config.config, flush=True)
                return False
            elif not isinstance(cpu_grad, paddle.Tensor) or not isinstance(custom_grad, paddle.Tensor):
                print(f"[gradient {i} type error]", self.api_config.config, flush=True)
                return False
            
            try:
                # Handle bfloat16 data type
                if cpu_grad.dtype == paddle.bfloat16:
                    cpu_grad = paddle.cast(cpu_grad, dtype="float32")
                if custom_grad.dtype == paddle.bfloat16:
                    custom_grad = paddle.cast(custom_grad, dtype="float32")
                
                # Convert CustomDevice gradient to CPU for comparison
                custom_grad_cpu = custom_grad.cpu()
                
                # Use precision comparison with tolerance 1e-2
                self.np_assert_accuracy(cpu_grad.numpy(), custom_grad_cpu.numpy(), 1e-2, 1e-2)
                
            except Exception as err:
                print(f"[gradient {i} accuracy error]", self.api_config.config, "\n", str(err), flush=True)
                return False
        
        return True
    
    def test(self):
        """Main test function"""
        
        # 1. Skip APIs that don't need testing
        if self.need_skip():
            print("[Skip]", flush=True)
            return
        
        # 2. Check if CustomDevice is available
        if not self.check_custom_device_available():
            print(f"[{self.custom_device_type} not available]", self.api_config.config, flush=True)
            write_to_log("custom_device_not_available", self.api_config.config)
            return
        
        # 3. Parse Paddle API information
        if not self.ana_paddle_api_info():
            print("ana_paddle_api_info failed", flush=True)
            return
        
        # 4. Generate Numpy input data
        try:
            if not self.gen_numpy_input():
                print("gen_numpy_input failed")
                return
        except Exception as err:
            print("[numpy error]", self.api_config.config, "\n", str(err))
            write_to_log("numpy_error", self.api_config.config)
            return
        
        # 5. Run API on CPU (including forward and backward)
        cpu_output, cpu_grads = self.run_on_device("cpu", 0)
        if cpu_output is None:
            print("[cpu execution failed]", self.api_config.config, flush=True)
            write_to_log("cpu_error", self.api_config.config)
            return
        
        # 6. Run API on CustomDevice (including forward and backward)
        custom_output, custom_grads = self.run_on_device(self.custom_device_type, self.custom_device_id)
        if custom_output is None:
            print("[custom device execution failed]", self.api_config.config, flush=True)
            write_to_log("custom_device_error", self.api_config.config)
            return
        
        # 7. Compare forward results
        print("[forward test begin]")
        forward_pass = self.compare_outputs(cpu_output, custom_output)
        
        # 8. Backward test (if needed)
        backward_pass = True
        if self.need_check_grad():
            print("[Backward test begin]")
            
            # Check CPU backward results
            if cpu_grads is None:
                print("[cpu backward execution failed]", self.api_config.config, flush=True)
                write_to_log("cpu_backward_error", self.api_config.config)
                backward_pass = False
            # Check CustomDevice backward results
            elif custom_grads is None:
                print("[custom device backward execution failed]", self.api_config.config, flush=True)
                write_to_log("custom_device_backward_error", self.api_config.config)
                backward_pass = False
            else:
                # Compare gradient results
                backward_pass = self.compare_gradients(cpu_grads, custom_grads)
        else:
            # When backward test is not needed, pass by default
            backward_pass = True
        
        # 9. Final result judgment
        if forward_pass and backward_pass:
            print("[Pass]", self.api_config.config, flush=True)
            write_to_log("pass", self.api_config.config)
        else:
            print("[Fail]", self.api_config.config, flush=True)
            if not forward_pass:
                write_to_log("accuracy_error", self.api_config.config)
            elif not backward_pass:
                write_to_log("accuracy_error", self.api_config.config)