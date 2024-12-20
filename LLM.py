from vllm import LLM as vLLM
from vllm import SamplingParams

class LLM:
    def __init__(self, model_path="./Qwen2.5-1.5B-Instruct"):
        """初始化LLM模型
        
        Args:
            model_path: 模型路径，默认为当前目录下的Qwen2.5-1.5B-Instruct
        """
        self.model = vLLM(
            model=model_path,
            trust_remote_code=True,
            dtype="float16",  # 使用float16以减少显存占用
            gpu_memory_utilization=0.8,  # 控制GPU显存使用率
            max_num_batched_tokens=256,  # 控制每批处理的最大token数
            max_num_seqs=1  # 限制并发序列数为1，相当于batch size=1
        )
        
        # 设置采样参数
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=2048,
            max_input_tokens=256  # 设置输入token长度上限为256
        )
        
    def generate(self, prompt):
        """生成回复
        
        Args:
            prompt: 输入提示词
            
        Returns:
            str: 模型生成的回复
        """
        outputs = self.model.generate(prompt, self.sampling_params)
        response = outputs[0].outputs[0].text
        return response.strip()
    
    def __call__(self, prompt):
        """方便直接调用对象生成回复"""
        return self.generate(prompt)

if __name__ == "__main__":
    # 测试代码
    llm = LLM()
    prompt = "你好，请介绍一下自己。"
    response = llm(prompt)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
