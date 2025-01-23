from dotenv import load_dotenv
load_dotenv()

import os
import base64
import re
import requests
import torch
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
#from transformers import AutoTokenizer, AutoModelForCausalLM
from diffusers import StableDiffusionPipeline

API_URL = "https://api-inference.huggingface.co/models/dima806/facial_emotions_image_detection"
headers = {"Authorization": "Bearer {os.getenv('HUGGINGFACE_API_TOKEN')}"}

# 定义情绪标签到表情符号的映射
EMOTION_TO_EMOJI = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😄",
    "neutral": "😐",
    "sad": "😢",
    "surprise": "😲"
}

def query(image_data):
    response = requests.post(API_URL, headers=headers, data=image_data)
    return response.json()

@csrf_exempt
def capture_image(request):
    if request.method == 'POST':
        image_data = request.POST.get('image')
        if not image_data:
            return render(request, 'camera.html', {'error': 'No image data received'})
        
        try:
            # 解码 base64 图像数据
            image_data = re.sub('^data:image/.+;base64,', '', image_data)
            image_data = base64.b64decode(image_data)
            
            # 临时保存图像
            file_name = 'captured_image.png'
            file_path = default_storage.save(file_name, ContentFile(image_data))
            
            # 调用情绪检测 API
            result = query(image_data)
            
            # 删除临时文件
            default_storage.delete(file_path)
            
            # 将 JSON 结果转换为带有表情符号的文本
            if 'error' in result:
                result_text = result['error']
            else:
                # 提取所有情绪标签及其对应的表情符号
                result_text = ', '.join([f"{EMOTION_TO_EMOJI.get(emotion['label'], emotion['label'])}: {emotion['score']:.2f}" for emotion in result])
            
            # 使用 deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B 模型生成描述文本
            tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
            model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
            input_text = f"The detected emotions are {result_text}"
            inputs = tokenizer(input_text, return_tensors="pt")
            outputs = model.generate(inputs.input_ids, max_length=50)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 使用 stable-diffusion-v1-5 模型生成图像
            pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
            pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
            image = pipe(generated_text).images[0]
            
            # 将生成的图像保存为文件
            image_file_path = default_storage.save('generated_image.png', ContentFile(image.tobytes()))
            
            # 将图像数据传递回模板
            image_data_url = f"data:image/png;base64,{base64.b64encode(image_data).decode()}"
            generated_image_url = f"data:image/png;base64,{base64.b64encode(image.tobytes()).decode()}"
            
            return render(request, 'camera.html', {
                'message': 'Image uploaded successfully',
                'result': result_text,
                'generated_text': generated_text,
                'image_data_url': image_data_url,
                'generated_image_url': generated_image_url
            })
        except ValueError:
            return render(request, 'camera.html', {'error': 'Invalid image data format'})
    return render(request, 'camera.html')

@csrf_exempt
def capture_audio(request):
    if request.method == 'POST':
        audio_file = request.FILES.get('audio')
        if not audio_file:
            return JsonResponse({'error': 'No audio file received'})

        try:
            # 保存音頻文件
            audio_path = default_storage.save('temp.wav', ContentFile(audio_file.read()))

            # 這裡可以添加處理音頻文件的代碼，例如情緒分析等

            # 刪除臨時音頻文件
            default_storage.delete(audio_path)

            return JsonResponse({'success': 'Audio captured successfully'})
        except Exception as e:
            return JsonResponse({'error': str(e)})
    return render(request, 'audio.html')