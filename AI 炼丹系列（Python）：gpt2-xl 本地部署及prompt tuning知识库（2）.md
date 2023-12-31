## 问题：
- gpt2-xl的本地部署
- prompt中上下文的zero-shot, one-shot对gpt2生成文本的影响
- langchain如何实现知识库的prompt tuning 
- 如何进行Adapter Tuning
## 编写代码
引入上一篇文章中定义好的gpt2-xl接口，并继承自langchain.llms.base.LLM实现本地大模型的自定义类。

```
class APIPathEnum(str, Enum):

    simple_url = "http://127.0.0.1:8088/ai/gpt2xl_gc"
    content_url = "http://127.0.0.1:8088/ai/gpt2xl_embedding_gc"
    embedding_url = "http://127.0.0.1:8088/ai/gpt2xl_embedding"

```
langchain库中的继承关系，采用的是Callable的方式实现，所以子类要实现基类的_call方法，它会在基类的__call__方法中被调用。

```
class LocalGPT2XL(LLM):
    
    def __init__(self):
        super().__init__()

    
    def _llm_type(self) -> str:
        return "gpt2xl_model"

    
    def _invoke_api(self, url:str, prompt:str, content:list[str]|None = None):
        req_dict = {"text": prompt}
        if content is not None:
            req_dict["content"] = content
        # 构建json请求
        headers = {"Context_Type": "application/json"}
        resp = requests.post(url, json=req_dict, headers= headers)
        # 获取json结果
        if resp.status_code == 200:
            return resp.json()["gpt2xl"]
        else:
            return None
        

    def _call(self, prompt: str, stop: List[str] | None = None, run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any) -> str:
        # 实现基类方法，调用两个接口
        if kwargs is None or kwargs["content"] is None:
            resp = self._invoke_api(APIPathEnum.simple_url.value, prompt)
        else:
            resp = self._invoke_api(APIPathEnum.content_url.value, prompt, content = kwargs["content"])
        # 确保有返回结果
        return  resp if resp is not None else "gpt2xl_error"
```
继承自langchain_core.embeddings.Embeddings的自定义嵌入编码类，以便实现本地矢量库的存储和查询。

```
class LocalGPT2Embedding(Embeddings):

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
    
    # 调用本地嵌入编码接口
    def _invoke_api_once(self, txt:str):
        req_dict = {"text": txt}
        headers = {"Context_Type": "application/json"}
        resp = requests.post(APIPathEnum.embedding_url.value, json=req_dict, headers= headers)
        if resp.status_code == 200:
             return resp.json()["data"]
        else:
             return None
    
    # 实现基类方法
    def embed_query(self, text: str) -> List[float]:
        resp = self._invoke_api_once(text)
        return resp
    
    # 实现基类方法
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embedding_list = []
        for text in texts:
            resp = self._invoke_api_once(text)
            embedding_list.append(resp)
        return embedding_list
```
langchain.vectorstores中定义了各类矢量库的调用方法，比如 Chroma, FAISS, Pinecone等，都是继承自langchain_core.vectorstores.VectorStore以便提供统一的操作界面，也可以引入矢量库厂商提供的类库进行增强，比如Pinecone, 可参考相关文档。

首先定义好本地知识文件和矢量库的存储路径：

```
class DataPathEnum(str, Enum):
    LANGCHAIN_DATA_DIR = "knowledge"
    LANGCHAIN_DATA_VECTOR_DIR = "knowledge/vector_store"
    
    def __str__(self):
        return os.path.join(cur_path, "data", self.value)
```
目录结构如下：
![目录.jpg](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/1936dbf9519a46229fc4f2b4527acdb1~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=361&h=82&s=8175&e=jpg&b=191919)

矢量库可以载入各类文件，包括图片，PDF等。在这里只处理.txt文件，具体内容可以是各类知识问答等。以便提供用户问题相关度高的附加上下文。

比如，mycontent.txt中：

```
who am i? I am YS-SRT's Robot
who are you? I am your robot, mr. YS-SRT
```
mycontent2.txt中:

```
How old are you? I am six year old
what's you age? I am one year older then you, you are five years old
```
然后实现本地矢量库的载入和知识类文件的导入。这里要特别提到矢量计算文本相关度的几个策略，FAISS提供了五个策略，其中DistanceStrategy.COSINE 余弦，较常用。默认的是DistanceStrategy.EUCLIDEAN_DISTANCE 欧几里得几何距离，剩下的还有：

```
    MAX_INNER_PRODUCT = "MAX_INNER_PRODUCT"
    DOT_PRODUCT = "DOT_PRODUCT"
    JACCARD = "JACCARD"
```
不幸的是OpenAI使用的是unit normed，和其它大多不同。所以上述策略可以说是对gpt2产生的embedding矢量的计算无用。好在可以自定义矢量计算相关度的函数，慢慢摸索较契合的算法。甚至可以直接设定score_threshold来过滤查询后的结果。
```
VDB_SIGN = "myknowledge"
VDB_DIR = str(DataPathEnum.LANGCHAIN_DATA_VECTOR_DIR)

class LocalVectorStore():
   # 加载本地矢量库
   @classmethod
   @lru_cache
   def init_vector_db(cls, embedding: Embeddings) -> None:
       if os.path.exists(os.path.join(VDB_DIR, VDB_SIGN + ".faiss")):
           return FAISS.load_local(VDB_DIR, embedding, VDB_SIGN, distance_strategy=DistanceStrategy.COSINE)
           # return Chroma(VDB_SIGN,embedding,VDB_DIR) Chroma矢量库的文件后缀有差异
       else:
           return cls.load_docs_into_db(embedding)
       
   # 导入本地所以.txt文件内容，期间会自动调用Embedding的接口方法产生矢量值并保存下来
   @classmethod
   def load_docs_into_db(self, embedding)->FAISS:
       files = map(lambda f: os.path.join(dir, f), os.listdir(VDB_DIR))
       files = filter(lambda f: os.path.isfile(f) and f.endswith(".txt"), files)
       
       docs = []
       for f in files:
           text_splitter = CharacterTextSplitter()
           docs += TextLoader(f, "utf8").load_and_split(text_splitter)

       vector_db = FAISS.from_documents(docs, embedding)
       vector_db.save_local(VDB_DIR, VDB_SIGN)

       # vector_db = Chroma.from_documents(docs, embedding)
       # vector_db.persist()

       return vector_db
       
```
继承自langchain.chains.LLMChain来实现自定义的Chain, 按langchain库的惯例，依然是实现其基类的_call方法。

```
QA_TEMPLATE="""QA Processing: >>>
{qa_history}
Q:{question}
A:"""

class LocalKnowledgeQAChain(LLMChain):
    # 初始化本地矢量库，传入自定义Embedding嵌入编码接口，
    # 会在本地目录下生成myknowledge.faiss和mykonwledge.pki文件
    local_vdb = LocalVectorStore.init_vector_db(LocalGPT2Embedding())
    
    # 初始化时设置自定义PromptTemplate
    def __init__(self) -> None:
        qa_prompt = PromptTemplate(input_variables=["question"], 
                                   template=QA_TEMPLATE)
        super().__init__(llm=LocalGPT2XL(),
                         memory=ConversationBufferMemory(memory_key="qa_history"),
                         prompt=qa_prompt,
                         output_key="gpt2xl",
                         return_final_only=True,
                         verbose=True)
    # 重写基类_call方法
    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        content=None
        # 本地矢量库查询,使用矢量相关度计算
        related_contents = self.local_vdb._similarity_search_with_relevance_scores(inputs["question"])
        if related_contents is not None:
            content = [doc.page_content for doc, _ in related_contents]
        # 其他相关库查询，比如Database，Elasticsearch等
        other_contents = self._search_from_sources(inputs)
        if other_contents is not None:
            content = content + other_contents if content is not None else other_contents
        # 调用自定义大模型的方法，最终会调用自己实现的_call方法
        resp = self.llm.generate(prompts=[self._build_template(inputs)], content=content)
        return self.create_outputs(resp)[0] #基类提供的结果解析方法
        
    # 利用基类方法组合PromptTemplate
    def _build_template(self, inputs: Dict[str, Any]):
        prompts, _ = self.prep_prompts([inputs])
        return prompts[0].to_string()
        
    # 其他数据源的查询
    def _search_from_sources(self, inputs: Dict[str, Any]) ->list[str]:
        pass # from DB or ES

    # 跳过所有继承机制直接调用本地gpt2接口，作为结果对比方法
    def ask(self, question:str):
        related_contents = self.local_vdb.similarity_search_with_score(question)
        if related_contents is not None:
            content = [doc.page_content for doc, _ in related_contents]
            
        other_contents = self._search_from_sources(inputs)
        if other_contents is not None:
            content = content + other_contents if content is not None else other_contents
            
        # 调用自定义大模型_call方法
        resp = self.llm._call(prompt=question, content=content)
        return resp
```
写个ipynb来调用自定义链LocalKnowledgeQAChain中的方法，注意要加载项目目录，不然会提示找不到相关模块：

```
from aimodel.gpt2_langchain.gpt2_chain import LocalKnowledgeQAChain
chain = LocalKnowledgeQAChain()

result1 = chain.run(question="who am i?")
result1

result2 = chain.predict(question="who is he?")
result2

result3 = chain.ask("who is he?")
result3

```
下一篇谈谈如何微调gpt2模型和相关的基础知识，包含对Tensorflow, Pytoch库的熟练使用。

## 完整代码地址
![2023-11-27_093952.jpg](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/b3aa851968634d40a5ae82a5c9d3daf4~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=987&h=570&s=83093&e=jpg&b=fcfcfc)