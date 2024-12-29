# PosionedRAG简单版实现
论文链接：https://arxiv.org/abs/2402.07867  
笔记链接：https://q1mf21tt1eq.feishu.cn/wiki/I4VlwEQf7i0ovyk5LoRcWg8PnMd?from=from_copylink
![image](https://github.com/user-attachments/assets/87dde0b5-0c76-43eb-a24a-df8cc3bfb3c6)

## 对目标查询进行编码
```
query = "Who is the CEO of openai?"
query_input = tokenizer(query, padding=True, truncation=True, return_tensors="pt")
query_embedding = contriever(**query_input).detach().numpy()
```
## 制作恶意文本并插入语料库
```
mistake_answer = "Jonathan Allen"
for i in range(5):
    mis_text = dev_text(query, mistake_answer)
    poison_text = query + mis_text
    sentences.append(poison_text)
```
## 对语料库进行编码
```
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
txt_embeddings = contriever(**inputs).detach().numpy()
```
## 计算相似度，找到前k个最相关的文本
```
similarity_scores = [cosine_similarity(query_embedding, doc_embedding) for doc_embedding in txt_embeddings]
similarity_scores_array = np.array(similarity_scores).squeeze()

k = 5
top_k_indices = np.argsort(similarity_scores_array)[-k:][::-1]
```
## 将前k个文本和query制作prompt输入LLM
```
input_prompt = wrap_prompt(query, retrieve_text, k)

response = get_res(input_prompt)
print(response)
```

![image](https://github.com/user-attachments/assets/88d39048-61f2-4f85-bbd0-b92d8fcaf304)
