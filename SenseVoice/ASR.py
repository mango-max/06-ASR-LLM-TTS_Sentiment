import os
import librosa
import soundfile as sf
import numpy as np
import requests
from pynput import keyboard
import sounddevice as sd
# import time

def send_audio_files(files):
    url = "http://127.0.0.1:8866/api/v1/asr"
    
    # start_time = time.time()
    try:
        with open(files, "rb") as audio_file:
            files_data = [
                ("files", (files, audio_file, "audio/wav")),
            ]
            data = {
                "keys": "string",
                "lang": "zh"
            }
            
            response = requests.post(
                url, 
                headers={"accept": "application/json"}, 
                files=files_data, 
                data=data,
                timeout=5
            )
            
            response.raise_for_status()
            # end_time = time.time()
            # print(f"ASR请求耗时: {end_time - start_time:.2f}秒")
            return response.json()
            
    except requests.exceptions.Timeout:
        print("请求超时")
        return None
    except requests.exceptions.RequestException as e:
        print(f"请求失败：{e}")
        return None
    except Exception as e:
        print(f"发生错误：{e}")
        return None

def fill_size_wav(file_path):
    """修正 WAV 文件的头部大小信息，使其符合 WAV 格式规范。"""
    try:
        with open(file_path, "r+b") as f:
            size = os.path.getsize(file_path) - 8
            f.seek(4)
            f.write(size.to_bytes(4, byteorder='little'))
            f.seek(40)
            f.write((size - 36).to_bytes(4, byteorder='little'))
            f.flush()
    except Exception as e:
        print(f"修复 WAV 文件大小出错：{e}")

def process_voice(file_path):
    """处理接收到的语音文件，并使用 ASR 服务将其转换为文本。"""
    try:
        # start_total = time.time()
        
        # 修复WAV文件
        # start_time = time.time()
        fill_size_wav(file_path)
        # print(f"修复WAV文件耗时: {time.time() - start_time:.2f}秒")
        
        # 加载音频
        # start_time = time.time()
        y, sr = librosa.load(file_path, sr=None, mono=False)
        # print(f"加载音频耗时: {time.time() - start_time:.2f}秒")
        
        # 音频处理
        # start_time = time.time()
        y_mono = librosa.to_mono(y)
        y_mono = librosa.resample(y_mono, orig_sr=sr, target_sr=16000)
        sf.write(file_path, y_mono, 16000)
        # print(f"音频处理耗时: {time.time() - start_time:.2f}秒")
        
        # ASR识别
        text = send_audio_files(file_path)
        
        # print(f"总处理耗时: {time.time() - start_total:.2f}秒")
        return text
    except Exception as e:
        print(f"处理语音出错：{e}")
        return ""

class AudioRecorder:
    def __init__(self):
        self.recording = False
        self.frames = []
        self.sample_rate = 16000
        self.buffer_size = 1024
        # self.start_time = None

    def start_recording(self):
        if not self.recording:
            self.recording = True
            self.frames = []
            # self.start_time = time.time()
            print("录音开始...")
            self.stream = sd.InputStream(
                callback=self.audio_callback,
                samplerate=self.sample_rate,
                channels=1,
                blocksize=self.buffer_size
            )
            self.stream.start()

    def stop_recording(self, file_path='output.wav'):
        if self.recording:
            self.recording = False
            # record_duration = time.time() - self.start_time
            # print(f"录音时长: {record_duration:.2f}秒")
            
            # start_time = time.time()
            self.stream.stop()
            self.stream.close()
            
            audio_data = np.concatenate(self.frames)
            sf.write(file_path, audio_data, self.sample_rate)
            
            # print(f"保存录音耗时: {time.time() - start_time:.2f}秒")
            
            self.frames = []
            return file_path

    def audio_callback(self, indata, frames, time, status):
        """录音数据回调函数。"""
        if self.recording:
            self.frames.append(indata.copy())

def on_press(key):
    """键盘按下事件处理。"""
    try:
        if key.char == 'z':  # 按下 'z' 键开始录音
            recorder.start_recording()
    except AttributeError:
        pass

def on_release(key):
    """键盘释放事件处理。"""
    if key == keyboard.Key.esc:
        # 停止监听并优雅地退出程序
        print("退出程序...")
        return False
    try:
        if key.char == 'z':  # 释放 'z' 键停止录音
            file_path = recorder.stop_recording()
            print("处理语音...")
            text = process_voice(file_path)
            print("识别的文本：", text)
            print("\n按住Z开始语音输入，松开Z结束语音输入，按ESC退出程序")
    except AttributeError:
        pass

def check_server_device():
    """检查服务器使用的推理设备"""
    try:
        url = "http://127.0.0.1:8866/api/v1/device_info"  # 需要在服��器端添加此接口
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            device_info = response.json()
            print(f"服务器推理设备: {device_info['device']}")
    except Exception as e:
        print(f"无法获取服务器设备信息: {e}")

if __name__ == "__main__":
    recorder = AudioRecorder()
    # 添加设备信息检查
    # check_server_device()
    print("按住Z开始语音输入，松开Z结束语音输入，按ESC退出程序")
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
