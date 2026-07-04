import torch
from flask import Flask, request, jsonify
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, AutoTokenizer
import base64
from io import BytesIO
from PIL import Image
import traceback
from qwen_vl_utils import process_vision_info

VLM_MODEL_ID = '/data/models/Qwen2.5-VL-7B-Instruct'
LLM_MODEL_ID = '/data/models/Qwen2.5-7B-Instruct'

# 从环境变量获取GPU设置
# 当设置了CUDA_VISIBLE_DEVICES后，第一个可见的GPU总是cuda:0
# 例如: CUDA_VISIBLE_DEVICES=3 时，cuda:0 实际对应物理GPU 3
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

print(f"正在使用设备: {DEVICE}")
print(f"正在加载 VLM 模型: {VLM_MODEL_ID}，请耐心等待...")

vlm_processor = AutoProcessor.from_pretrained(VLM_MODEL_ID, trust_remote_code=True)
vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    VLM_MODEL_ID,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).to(DEVICE).eval()

print(f"VLM 模型加载成功！")
print(f"正在加载 LLM 模型: {LLM_MODEL_ID}，请耐心等待...")

llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID, trust_remote_code=True)
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_ID,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).to(DEVICE).eval()

print("所有模型加载成功！服务已准备就绪。")

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

@app.route('/chat', methods=['POST'])
def chat():
    if not request.is_json:
        return jsonify({"error": "请求必须是 JSON 格式"}), 400

    data = request.get_json()
    prompt = data.get('prompt')
    image_b64 = data.get('image')
    print(f'成功接收到prompt: {prompt}')

    if not prompt:
        return jsonify({"error": "请求体中缺少 'prompt' 字段"}), 400

    try:
        if image_b64:
            print("接收到图片，使用 VLM 模型处理多模态请求。")

            image_bytes = base64.b64decode(image_b64)
            image = Image.open(BytesIO(image_bytes)).convert('RGB')

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            text = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = vlm_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(vlm_model.device)

            with torch.no_grad():
                outputs = vlm_model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                )

            response_text = vlm_processor.batch_decode(
                outputs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            response_text = response_text.split("assistant\n")[-1] if "assistant\n" in response_text else response_text

            del image, image_bytes
            del messages
            del text, image_inputs, video_inputs
            del inputs, outputs
            torch.cuda.empty_cache()

        else:
            print("未接收到图片，使用专门的 LLM 模型处理纯文本请求。")

            messages = [{"role": "user", "content": prompt}]

            text = llm_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = llm_tokenizer(
                [text],
                return_tensors="pt",
                padding=True
            ).to(llm_model.device)

            with torch.no_grad():
                outputs = llm_model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                )

            input_length = inputs.input_ids.shape[1]
            response_ids = outputs[0][input_length:]
            response_text = llm_tokenizer.decode(response_ids, skip_special_tokens=True)

            del messages, text
            del inputs, outputs, response_ids
            torch.cuda.empty_cache()

        print(f"成功生成响应: {response_text[:80]}...")
        return jsonify({"response": response_text})

    except Exception as e:
        print(f"处理请求时发生错误: {e}")
        traceback.print_exc()
        return jsonify({"error": f"服务器内部错误: {str(e)}"}), 500

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='LLM/VLM Server')
    parser.add_argument('--port', type=int, default=5001, help='服务器端口 (默认: 5001)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='服务器地址 (默认: 0.0.0.0)')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"🚀 LLM/VLM Server 启动中...")
    print(f"{'='*60}")
    print(f"  地址: {args.host}:{args.port}")
    print(f"  设备: {DEVICE}")
    print(f"  VLM模型: {VLM_MODEL_ID}")
    print(f"  LLM模型: {LLM_MODEL_ID}")
    print(f"{'='*60}\n")

    # 建议使用 waitress 或 gunicorn 等生产级 WSGI 服务器替代 app.run
    app.run(host=args.host, port=args.port)
