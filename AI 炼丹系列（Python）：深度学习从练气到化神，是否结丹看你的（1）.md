## 问题：
1. 深度学习训练的常规步骤
2. 微调模型的几个思考方向
3. 构建自己的数据集用于微调大模型
4. 各种微调方式的实现

## 编写代码
先用一个简单的中文手写识别的深度学习例子来说明训练的过程，这里分别使用PyTorch和TenserFlow来实现，以便比较两个工具库的不同风格。

**Talk is cheap， Show code.**

数据存放的路径：总共15000张图片，可以去kaggle.com下载 https://www.kaggle.com/datasets/gpreda/chinese-mnist/data

![手写.jpg](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/f839073ae59249799aa77bbbf92d0ab5~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=205&h=227&s=20964&e=jpg&b=181818)

图片文件名最后一个数字和图片内容中的文字对应关系是：

["零一二三四五六七八九十百千万亿"] -> index + 1

按咱的惯例，定义数据路径：
```
cur_path = os.getcwd()

class DataPathEnum(str, Enum):
    ZH_HANDWRITTING_IMG_DIR = "chinese_handwritting/images"
    GPT2XL_TRAIN_DATA_DIR = "gpt2xl"
    MODEL_CHECKPOINT_DIR = "checkpoints"

    def __str__(self):
        return os.path.join(cur_path, "data", self.value)
```

然后定义两个类库都可以使用的数据基类：
```
IMAGE_DIR = str(DataPathEnum.ZH_HANDWRITTING_IMG_DIR)

# image = 64 * 64
class HWData():

    def __init__(self) -> None:
        self.image_files = os.listdir(IMAGE_DIR)
        self.character_str ="零一二三四五六七八九十百千万亿"
        self.image_folder:str = IMAGE_DIR
        
    # 获取图片路径和标签
    def get_image_path_vs_lable(self, index):
        image_file = self.image_files[index]
        image_path = os.path.join(self.image_folder, image_file)
        label = int(image_file.split(".")[0].split("_")[-1]) -1
        return label, image_path
        
    # 在ipynb文件展示图片并显示标签，以便和预测的结果进行比较
    def plot_image(self, index):
        label, image_path = self.get_image_path_vs_lable(index)
        image = Image.open(image_path)

        plt.title("label: " + str(label) + "/" + self.character_str[label])
        plt.imshow(image)
```
先用PyTorch库来操作, 继承自torch.utils.data.Dataset实现自定义数据集，引入图像处理torchvision库。
```
class HWDataset(HWData, Dataset):

    def __init__(self) -> None:
        super().__init__()
        self.transform = torchvision.transforms.ToTensor()
    
    # 实现基类方法
    def __len__(self):
        return len(self.image_files)
        
    # 实现基类方法
    def __getitem__(self, index) -> Any:
        label, image_path = self.get_image_path_vs_lable(index)
        image_tensor = self.transform(Image.open(image_path))
        
        # 标签向量，独热模式
        target = torch.zeros((15))
        target[label] = 1.0

        return image_tensor, target, self.character_str[label]
    
    # 随机抽一张图，做验证比较
    def get_random_item(self):
        index = randint(0, self.__len__() -1)
        return self.__getitem__(index), index
```
继承自torch.nn.Module实现模型类：
```
class TorchClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 这里只构建全连接层，以便和后面加了卷积和池化层的模型进行比较
        self.model = nn.Sequential(
            nn.Linear(64*64, 1000), # 图片格式64x64
            nn.LeakyReLU(0.02),
            nn.Linear(1000, 100),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(100),
            nn.Linear(100, 15),
            nn.Softmax(dim = 1)
        )
        # 定义优化函数
        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.001)
        # 定义损失函数
        self.loss = nn.BCELoss()
        # 初始化数据集
        self.dataset = HWDataset()
        # 训练过程数据记录
        self.counter = 0
        self.progress = []
        # 模型参数保存路径
        self.checkpoint_file = os.path.join(str(DataPathEnum.MODEL_CHECKPOINT_DIR),
                                            "zh_hw_torch.pth")
    # 实现基类方法
    def forward(self, x):
        # 全连接层接受扁平的批次数据 
        x = x.view(x.size(0), -1) # 等同 nn.flatten(x)
        return self.model(x)
    
    # 训练方法
    def _train(self, x, y):
        outputs = self.forward(x)
        loss = self.loss(outputs, y)

        # 优化过程
        self.optimiser.zero_grad() #重置，否则在下个批次梯度会叠加
        loss.backward() #反向传播
        self.optimiser.step() #调整每层的参数
        
        # 记录过程数据(损失函数的值)
        self.counter += 1
        if(self.counter % 10 == 0):
            self.progress.append(loss.item())
    
    # 按周期次数进行训练
    def train(self, epochs: int):
        print('start train model...')
        for epochs in range(epochs):
            data_loader = DataLoader(self.dataset, batch_size=100, shuffle=True)
            for index, (data, target, target_char) in enumerate(data_loader):
                self._train(data, target)
                
        self._plot_progress() #绘制训练过程中损失函数的值

    # 随机抽一张图，验证查看
    def random_eval_model(self):
        (data, target, _), index = self.dataset.get_random_item()
        
        self.dataset.plot_image(index)
        with torch.no_grad():
            output = self.forward(data)
        
        df = pd.DataFrame(output.detach().numpy()).T
        df.plot.barh(rot=0, legend=False, ylim=(0, 15), xlim=(0,1))
        
    def _plot_progress(self):
        df = pd.DataFrame(self.progress, columns = ["loss"])
        df.plot(title="counter:" + str(self.counter), ylim=(0, 1.0), figsize=(16,8), 
                alpha=0.5, marker=".", grid=True, yticks=(0,0.25,0.5))     
    
    # 保存模型参数
    def save_model_state(self):
        torch.save(self.model.state_dict(), self.checkpoint_file)

```
写个ipynb文件来调用模型进行训练：
```
from aitrain.classifier_torch import HWDataset, TorchClassifier
#测试数据集
dataset = HWDataset()
dataset.plot_image(2)

# 训练模型并保存参数
classifier = TorchClassifier()
classifier.train(3)
classifier.save_model_state()
```
![torch_train.jpg](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/88518c3376a54f4fb105978789e20701~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=707&h=504&s=42625&e=jpg&b=fefefe)
```
# 随机验证并比较预测结果
classifier.random_eval_model()
```
![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/1612db9db0ca4cdb87039e38fd997ecf~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=586&h=941&s=35846&e=png&b=fefefe)

下面用TensorFlow2来实现相同的模块和功能，以便比较两种类库的操作区别。
继承自tensorflow.keras.utils.Sequence实现数据集
```
class TFDataset(HWData, tf.keras.utils.Sequence):
    def __init__(self, batch_size) -> None:
        super().__init__()
        self.batch_size = batch_size
        
    # 实现基类方法，这里返回的批次
    def __len__(self):
        return math.ceil(len(self.image_files)/self.batch_size)
        
    # 实现基类方法，依然是批次数据
    def __getitem__(self, idx):
        low = idx * self.batch_size
        high = min(low + self.batch_size, len(self.image_files))
        
        image_list = []
        label_list = []
        
        for index in range(low, high):
           label, image_path = self.get_image_path_vs_lable(index)
           image_array = np.array(list(Image.open(image_path).getdata()))
           image_list.append(image_array)

           # 标签的独热模式
           target =[0 if label != i else 1 for i in range(15)] 
           label_list.append(target)
           
        # 返回的是批次数据，注意shape与模型的输入参数一致
        return np.array(image_list).reshape(-1,64,64,1), np.array(label_list)
    
    # 实现基类方法，每个训练周期结束后调用
    def on_epoch_end(self):
        random.shuffle(self.image_files)
```
使用tensorflow.keras.Sequential构建TF模型，这里添加卷积和池化层，只是比较一下两个模型的预测准确率。
```
class TFClassifier():

    def __init__(self, data: TFDataset) -> None:
        model = tf.keras.Sequential()
        layers = tf.keras.layers
        
        model.add(layers.Rescaling(1./255)) # 压缩到0 ~ 1
        model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                            input_shape=(64, 64, 1)))
        model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
        model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
        model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(layers.Flatten()) #扁平层连接卷积层和全连接层
        model.add(layers.Dense(units=1024, activation='relu'))
        model.add(layers.Dropout(0.2)) # 丢弃一些特征
        model.add(layers.Dense(units=256, activation='relu'))
        model.add(layers.Dense(units=15, activation='softmax'))
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        self.model = model
        self.data = data
        self.checkpoint_file = os.path.join(str(DataPathEnum.MODEL_CHECKPOINT_DIR),
                                            "zh_hw_tf.h5")
    # 训练模型
    def train(self, epochs):
        his = self.model.fit(self.data, epochs=epochs, verbose=2).history # 
        self.__plot_history(his)
        self.model.summary() # 模型概要
    
    # 随机选图对比验证
    def random_eval_model(self):
        index = random.randint(0, len(self.data.image_files)-1)
        self.data.plot_image(index)

        label, image_path= self.data.get_image_path_vs_lable(index)
        image_array = np.array(list(Image.open(image_path).getdata())).reshape(-1, 64, 64, 1)
        prediction = self.model.predict(image_array)
        df = pd.DataFrame(prediction[0])
        df.plot.barh(rot=0, legend=False, ylim=(0, 15), xlim=(0,1))
        
    # 保存模型参数    
    def save_model_weights(self):
        self.model.save_weights(self.checkpoint_file)

    def load_model_weights(self):
        self.model.load_weights(self.checkpoint_file)   

    # 绘制历史数据，这里没有导入验证数据集，所以没有与验证相关损失函数值
    def __plot_history(self, history):
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].plot(history['loss'], label='training loss')
        # axs[0].plot(history['val_loss'], label='validation loss')
        axs[0].legend(loc='upper left')
        axs[0].set_title('training data vs validation data')

        axs[1].plot(history['accuracy'], label='testing accuracy')
        # axs[1].plot(history['val_accuracy'], label='validation accuracy')
        axs[1].set_ylim([0, 1])
        axs[1].legend(loc='upper left')
        axs[1].set_title('accuracy')

        axs.flat[0].set(xlabel='epochs', ylabel='loss')
        axs.flat[1].set(xlabel='epochs', ylabel='accuracy')

        plt.show()
```
写个ipynb文件来调用模型进行训练：
```
from aitrain.classifier_tensorflow import TFDataset, TFClassifier
# 验证数据集
data = TFDataset(100)
data.plot_image(5)

# 训练模型
classifier = TFClassifier(data)
classifier.train(3)
classifier.save_model_weights()
```
![tf_train.jpg](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/5a68cd9e6c504e9d8f31bd4daa688f07~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=689&h=660&s=78924&e=jpg&b=fcfcfc)
```
classifier.random_eval_model()
```
![tf_eval.jpg](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/0f0e902d7ec4430c91083631a28d67a4~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=569&h=992&s=48402&e=jpg&b=fefefe)

从构建数据集和模型的方式，可见TensorFlow和PyTorch封装的方式不同，但是都能方便地实现自己的想法。后面提到的transformers可以很好的融合这两个深度学习库。

从问题解决的角度看，与问题相适应的复杂度模型会很好地平衡训练时间和预测准确度，所以说优化方法和思维方式一直都是不断提高AI能力的推动力。

下一篇谈谈巨量参数的语言模型，如何在降低参数精度载入、增加AB低秩矩阵Tuning层来节省Full Fine-Tunning全量微调的计算资源需求。

## 完整代码地址
![my_github.jpg](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/a6b93e2ca4b94fd9b80c216fa8485284~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1226&h=457&s=66411&e=jpg&b=fefefe)