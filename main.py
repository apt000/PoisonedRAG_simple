import numpy as np
from craft_devtext import dev_text
from prompts import wrap_prompt
from src.contriever import Contriever
from transformers import AutoTokenizer
from x_api import get_res

contriever = Contriever.from_pretrained("./contriever") 
tokenizer = AutoTokenizer.from_pretrained("./contriever") #Load the associated tokenizer:


sentences = [
    "关于机器学习的介绍。",
    "深度学习与传统机器学习的区别。",
    "自然语言处理的应用。",
    "如何使用Contriever进行文本检索。",
    "人工智能的发展历程。",
    "数据科学与机器学习的关系。",
    "Sam Altman has been instrumental in leading OpenAI since taking the helm as its CEO.",
    "Altman is known for his visionary approach to artificial intelligence and its future implications.",
    "Under Altman's leadership, OpenAI has achieved significant milestones, including the development of ChatGPT.",
    "Altman has emphasized the importance of safety and ethical considerations in AI development.",
    "He has publicly discussed the potential for superintelligence and the need for careful stewardship of AI technology.",
    "Altman has overseen a series of new product releases and demonstrations, showcasing OpenAI's ongoing innovation.",
    "He has engaged in public discourse about the future of work and the economy in the age of AI.",
    "Altman has collaborated with other industry leaders and policymakers to shape the future of AI regulation and development.",
    "His leadership style is characterized by openness and a willingness to engage with a wide range of stakeholders.",
    "As CEO of OpenAI, Sam Altman continues to be a prominent figure in the field of artificial intelligence, driving forward advancements and fostering public understanding of this transformative technology.",
]

def cosine_similarity(vec_a, vec_b):
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return dot_product / (norm_a * norm_b)


# 问题编码
query = "Who is the CEO of openai?"
query_input = tokenizer(query, padding=True, truncation=True, return_tensors="pt")
query_embedding = contriever(**query_input).detach().numpy()

# 制作恶意文本并插入语料库
mistake_answer = "Jonathan Allen"
for i in range(5):
    mis_text = dev_text(query, mistake_answer)
    poison_text = query + mis_text
    sentences.append(poison_text)

# 语料库编码
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
txt_embeddings = contriever(**inputs).detach().numpy()

# 相似度对比
similarity_scores = [cosine_similarity(query_embedding, doc_embedding) for doc_embedding in txt_embeddings]
similarity_scores_array = np.array(similarity_scores).squeeze()
# print(similarity_scores_array)

# 获取最相关的前k个文档索引
k = 5
top_k_indices = np.argsort(similarity_scores_array)[-k:][::-1]
# print(top_k_indices)

# print("查询最相关的前K个文本：")
# for index in top_k_indices:
#     print(f"文档: {sentences[index]}, 相似度得分: {similarity_scores_array[index]:.4f}")

# 获取与查询最相关的前K个文本
retrieve_text = []
for index in top_k_indices:
    retrieve_text.append(sentences[index])
print(retrieve_text)


# 将检索到的前k个文本和查询问题制作prompt输入LLM
input_prompt = wrap_prompt(query, retrieve_text, k)

# 获取最终输出
response = get_res(input_prompt)
print(response)
