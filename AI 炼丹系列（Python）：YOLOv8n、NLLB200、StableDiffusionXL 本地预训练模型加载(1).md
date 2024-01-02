## 问题：
1、各种格式预训练模型的本地加载

2、基于预训练模型进行自己训练 
## 模型：
1、YOLOv8n, 物体识别模型 

2、NLLB200，语言翻译模型

3、StableDiffusionXL，文字图像生成模型

**Talk is cheap， Show code.**

## 编写代码
相关预训练模型可以去到[huggingface.co](https://huggingface.co/)网站下载，先建个模型的本地路径枚举，注意ml_models所在的目录与代码执行的入口文件在同一级:

模型保存的格式有多种，pt, bin, safetensors等
```
class ModelPathEnum(str, Enum):
    YOLOv8 = "ml_models/yolov8n.pt"
    NLLB200 = "ml_models/NLLB-200-600M"
    SDXL = "ml_models/stable-diffusion-xl-base-1.0/sd_xl_base_1.0_0.9vae.safetensors"
    SDXL_CONF="ml_models/stable-diffusion-xl-base-1.0/sd_xl_base.yaml"

    def __str__(self):
        return os.path.join(os.getcwd(), self.value)
```
YOLOv8n模型调用类:
```
class YOLO8Detection: # Local Model

    def load_model(self):
        self.model = YOLO(ModelPathEnum.YOLOv8)

    def attach_box_in_image(self, img: Image.Image)->Image:
         annotator = Annotator(np.array(img))
         predict = self.predict_single(img).get(0)
         predict.sort_values(by=['xmin'], ascending=True)

         for i, row in predict.iterrows():
             text = f"{row['name']}: {int(row['confidence']*100)}%"
             bbox = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
             annotator.box_label(bbox, text, color=colors(row['class'], True))
    
         return Image.fromarray(annotator.result())



    def predict_single(self, img:Image.Image):
        if not self.model:
            raise RuntimeError("model is not loaded")
        
        results = self.predict(img=img)

        predict_list = {}
        for index,result in enumerate(results):
            predict_bbox = pd.DataFrame(results[0].to("cpu").numpy().boxes.xyxy, columns=['xmin', 'ymin', 'xmax','ymax'])
            predict_bbox['confidence'] = results[0].to("cpu").numpy().boxes.conf
            predict_bbox['class'] = (results[0].to("cpu").numpy().boxes.cls).astype(int)
            predict_bbox['name'] = predict_bbox["class"].replace(results[0].names)

            predict_list[index] = predict_bbox
        
        return predict_list
        

    def predict(self, img: Image.Image | list) -> Results:
        return self.model.predict(source=img, conf=env.YOLO_DETECTION_MODEL_CONF,
                                    flipud=env.YOLO_DETECTION_MODEL_FLIPUD, 
                                    fliplr=env.YOLO_DETECTION_MODEL_FLIPLR,
                                    mosaic=env.YOLO_DETECTION_MODEL_MOSAIC)  
    
   # env.YOLO_DETECTION_MODEL_CONF default value is 0.5

```
FastAPI接口编写：

```
@aiAPI.post("/yolo_label")
async def detect_image_objects(img: UploadFile = File(...)) :
    image = Image.open(img.file)
    predict = ml_models["yolo_detection"].predict_single(image)
    return {"detect_result": json.loads(predict.get(0).to_json(orient="records"))}
    
@aiAPI.post("/yolo_label_show")
async def detect_image_objects(img: UploadFile = File(...)) :
    image = Image.open(img.file)
    predict_img = ml_models["yolo_detection"].attach_box_in_image(image)
    return StreamingResponse(content=img_to_bytes(predict_img), media_type="image/jpeg")
```

结果展示: 如果调高conf的值到0.8的话，识别率还不错

![yolov8n.jpg](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/f6af5cd23a6547beb0cba26bec013a98~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1068&h=1217&s=330830&e=jpg&b=edfaf4)

## 完整代码地址

![my_github.jpg](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/a6b93e2ca4b94fd9b80c216fa8485284~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1226&h=457&s=66411&e=jpg&b=fefefe)