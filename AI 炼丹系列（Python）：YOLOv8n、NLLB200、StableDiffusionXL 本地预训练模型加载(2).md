## 问题：

1、各种格式预训练模型的本地加载

2、基于预训练模型进行自己训练

## 模型：

1、YOLOv8n, 物体识别模型   [参考系列 1]

2、NLLB200，语言翻译模型

3、StableDiffusionXL，文字图像生成模型

**Talk is cheap， Show code.**

## 编写代码

NLLB200，翻译模型。这里设置中英文互译

```
class NLLB200Translator:

    def load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(ModelPathEnum.NLLB200)
        model = AutoModelForSeq2SeqLM.from_pretrained(ModelPathEnum.NLLB200)
        self.zh_en_translator = pipeline('translation', model=model, tokenizer=tokenizer,
                                         src_lang='zho_Hans', tgt_lang='eng_Latn', max_length=512)
        self.en_zh_translator = pipeline('translation', model=model, tokenizer=tokenizer,
                                         src_lang='eng_Latn', tgt_lang='zho_Hans', max_length=512)
        
    def en_to_zh(self, en_msg: str):
        return self.en_zh_translator(en_msg)
    
    def zh_to_en(self, zh_msg: str):
        return self.zh_en_translator(zh_msg)
```
接口方法

```
@aiAPI.post("/en2zh", response_model=TextPredict)
async def en2zh_translator(input:TextInput):
    result = ml_models["nllb_translator"].en_to_zh(input.text)
    return TextPredict(text= result[0].get("translation_text"))

@aiAPI.post("/zh2en", response_model=TextPredict)
async def en2zh_translator(input:TextInput):
    result = ml_models["nllb_translator"].zh_to_en(input.text)
    return TextPredict(text= result[0].get("translation_text"))
```

StableDiffusionXL，文字图像生成模型。SDXL依赖一些语言处理模型(xl-文字转图，xl_refine-按图扩展)，还可以附加一些约束模块。

在运行中会提示缺少的模块，比如： laion/CLIP-ViT-bigG-14-laion2B-39B-b160k。会到当前目录(实际加载时的目录，比如main.py运行时的目录)下寻找。

```
class Text2ImgBaseGenerator: #stable-diffusion-xl-1.0

    def load_model(self):
        pipe = StableDiffusionXLPipeline.from_single_file(ModelPathEnum.SDXL,
                                                        # torch_dtype=torch.float16, 
                                                        use_safetensors=True, 
                                                        variant="fp16",
                                                        original_config_file=ModelPathEnum.SDXL_CONF
                                                        )
        
        # pipe = pipe.to("cuda")
        pipe = pipe.to("cpu")
        pipe.enable_model_cpu_offload()
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
        self.pipe = pipe
    
    def predict(self, prompt:str) -> Image:
        return self.pipe(prompt).images[0]




class Text2ImgRefinerGenerator: #stable-diffusion-xl-1.0

    def load_model(self):
        pipe = StableDiffusionXLPipeline.from_single_file( ModelPathEnum.SDXL_REFINER, 
                                                                # torch_dtype=torch.float16, 
                                                                 variant="fp16", 
                                                                 use_safetensors=True)
        # pipe = pipe.to("cuda")
        pipe.enable_model_cpu_offload()
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
        self.pipe = pipe

    def predict_with_net(self, prompt:str, url:str) -> Image:
        ref_img = load_image(url)
        return self.predict(prompt, ref_img)

    def predict_with_img(self, prompt:str, ref_img:Image) -> Image:
        return self.pipe(prompt, image= ref_img.convert("RGB")).images
```
接口方法

```
@aiAPI.post("/sdxl_base_text2img")
async def sdxl_base_text2img(input:TextInput):
    predict_img = ml_models["sdxl_base"].predict(input.text)
    return StreamingResponse(content=img_to_bytes(predict_img), media_type="image/jpeg")
    
@aiAPI.post("/sdxl_refiner_img2img")
async def sdxl_Refiner_img2img(input:TextInput, img: UploadFile = File()):   
    predict_img = ml_models["sdxl_refiner"].predict_with_img(input.text, Image.open(img.file)             
    return StreamingResponse(content=img_to_bytes(predict_img), media_type="image/jpeg")
```
更多参数的使用，可以参考StableDiffusion-WebUI项目(https://github.com/AUTOMATIC1111/stable-diffusion-webui)

## 完整代码地址

![my_github.jpg](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/a6b93e2ca4b94fd9b80c216fa8485284~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1226&h=457&s=66411&e=jpg&b=fefefe)