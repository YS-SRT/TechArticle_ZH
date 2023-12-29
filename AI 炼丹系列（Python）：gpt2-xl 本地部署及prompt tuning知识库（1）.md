## 问题：
- gpt2-xl的本地部署
- prompt中上下文的zero-shot, one-shot对gpt2生成文本的影响
- langchain如何实现知识库的prompt tuning 
- 如何进行Adapter Tuning

## 编写代码
下载gpt2-xl模型的model.safetensors和tokenizer相关配置文件，添加到本地路径

```
class ModelPathEnum(str, Enum):
    YOLOv8 = "yolov8n.pt"
    NLLB200 = "NLLB-200-600M"
    Yi34BChat = "Yi-34b-Chat"
    SDXL = "stable-diffusion-xl-base-1.0/sd_xl_base_1.0_0.9vae.safetensors"
    SDXL_CONF="stable-diffusion-xl-base-1.0/sd_xl_base.yaml"
    SDXL_REFINER = "stable-diffusion-xl-base-1.0/sd_xl_refiner_1.0_0.9vae.safetensors"
    SDXL_REFINER_CONF="stable-diffusion-xl-base-1.0/sd_xl_refiner.yaml"
    GPT2XL = "gpt2-xl"

    def __str__(self):
        return os.path.join(os.getcwd(), "ml_models", self.value)
```
编写gpt2-xl模型加载类，特别提供embed接口，以便在本地实现embedding，无须调用其它在线API。gpt2采用的是text-embedding-**ada**-001，官方已经不建议使用。

**从output提取input的相关embedding, 并没有经过对比验证**，暂且认为本地环境下可以闭环进行相关技术的测试

```
class GPT2XLGenerator():
    
    def load_model(self):
        tokenizer = GPT2Tokenizer.from_pretrained(ModelPathEnum.GPT2XL, local_files_only=True)
        tokenizer.padding_side="left"
        model = GPT2LMHeadModel.from_pretrained(ModelPathEnum.GPT2XL, local_files_only=True, use_safetensors=True, pad_token_id=tokenizer.eos_token_id)
        model.eval() # 进入验证模式

        self.model = model
        self.tokenizer = tokenizer
        
    # 文本直接使用, zero-shot
    def predict(self, input_text:str):   
        return self._predict(input_text)

    # 使用模板提供上下文，one-shot/few-shot
    def predict_qa(self, input_text:str, input_content:list[str]):  
        contents = "\n".join([item for item in input_content])
        template = get_qa_template().replace("{question}", input_text).replace("{context}", contents)
        return self._predict(template)
        
    
    def _predict(self, input):
        count = len(input) 
        input_ids = self.tokenizer.encode(input, return_tensors='pt')
        with torch.no_grad(): #不自动计算梯度
            output = self.model.generate(input_ids, max_new_tokens=200, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
        output_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return output_text[count:-1].strip()  #截取生成的内容
    
    # 获取input embedding
    def get_input_embedding(self, input_text):
        input_ids = self.tokenizer(input_text, return_tensors='pt')
        with torch.no_grad():
            output = self.model(**input_ids)
        embedding = output[0][0].mean(dim=0).tolist() #输入在开始位置
        return embedding
```
提供调用接口

```
@aiAPI.post("/gpt2xl_gc")
async def gpt2xl_gc(input:TextInput):
    return {"gpt2xl": ml_models["gpt2xl"].predict(input.text)}

@aiAPI.post("/gpt2xl_embedding_gc")
async def gpt2xl_embedding_gc(input:TextInput2):
    return {"gpt2xl": ml_models["gpt2xl"].predict_qa(input.text, input.content)}

@aiAPI.post("/gpt2xl_embedding")
async def gpt2xl_embedding(input:TextInput):
    embeddings = ml_models["gpt2xl"].get_input_embedding(input.text)
    return EmbeddingsResponse(model="gpt2xl",data=embeddings)

```
gpt2对中文不太友好，所以只用英文进行相关测试。在问题之后携带相关度高的上下文一起提交给gpt2, 可以很好的调节回答内容的质量。

通常没有相关上下文的直接提问叫zero-shot，而one-shot就像LeetCode刷题时，时常会在题目中给个结果示例。

对比两个差别的输出，查看上下文对gpt2生成的影响：

zero-shot, 可见gpt2有点文艺范
![gpt2xl_zero_shot.jpg](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/6ecd3f3d3c3d47c2bbdf2258d8e86870~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1419&h=1254&s=250570&e=jpg&b=f3fcf8)

one-shot, 显然答案参考了上下文提示
![gpt2xl_one_shot.jpg](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/85d7af862f4e4a01a5eaa4458401c5a1~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1421&h=1279&s=248275&e=jpg&b=f3fcf8)

上下文模板用的是在网上最容易找到的文本，这是做题库AI Prompt Tuning所使用的，但gpt2还不足够智能做出恰当的反应。
```
# gpt2xl
PROMPT_TEMPLATE:str = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}
Question:{question}
"""
```

## 完整代码地址
![2023-11-27_093952.jpg](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/b3aa851968634d40a5ae82a5c9d3daf4~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=987&h=570&s=83093&e=jpg&b=fcfcfc)