import requests, base64
from io import BytesIO
import os

class FlaskFunc():
    def __init__(self, server_port=None):
        """
        初始化FlaskFunc

        Args:
            server_port: LLM server端口。如果为None，则根据CUDA_VISIBLE_DEVICES自动选择
        """
        if server_port is None:
            # 自动根据当前GPU选择对应的server端口
            cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
            # 取第一个GPU ID
            gpu_id = int(cuda_devices.split(',')[0]) if ',' in cuda_devices else int(cuda_devices)
            # 端口映射：GPU 0 → 5001, GPU 1 → 5002, ...
            server_port = 5001 + gpu_id

        self.server_url = f'http://127.0.0.1:{server_port}/chat'

    def get_llm_response(self,prompt):
        """
        通过调用本地 Flask API 获取纯文本模型的响应。
        """
        payload = {"prompt": prompt}
        response = None
        try:
            response = requests.post(self.server_url, json=payload, timeout=60)
            response.raise_for_status()  # 如果请求失败 (非 2xx 状态码), 会抛出异常
            result = response.json()["response"]

            response.close()
            del payload
            return result
        except requests.exceptions.RequestException:
            return "错误：无法连接到本地模型服务。"
        except KeyError:
            return f"[错误] 服务返回了无效的响应格式: {response.text}"
        finally:
            # 确保response被关闭
            if response is not None:
                response.close()

    def get_vlm_response(self, prompt, image):
        """
        通过调用本地 Flask API 获取视觉语言模型的响应。
        """
        buffered = None
        response = None
        try:
            # 将 PIL Image 对象转换为 Base64 编码的字符串
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            image_b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

            payload = {
                "prompt": prompt,
                "image": image_b64_str
            }

            # VLM 推理可能较慢，设置更长的超时时间
            response = requests.post(self.server_url, json=payload, timeout=120)
            response.raise_for_status() # 如果请求失败, 抛出异常
            result = response.json()["response"]

            response.close()
            buffered.close()
            del image_b64_str, payload

            return result

        except requests.exceptions.RequestException as e:
            print(f"[错误] 请求 VLM 服务失败: {e} (URL: {self.server_url})")
            return "错误：无法连接到本地模型服务。"
        except KeyError:
            return f"[错误] 服务返回了无效的响应格式: {response.text}"
        finally:
            # 确保资源被释放
            if buffered is not None:
                buffered.close()
            if response is not None:
                response.close()
