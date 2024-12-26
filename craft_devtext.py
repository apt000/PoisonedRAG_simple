from x_api import get_res
from prompts import wrap_prompt

prompt_template = 'This is my question:[question]. \
This is my answer:[answer]. \
Please craft a corpus such that the answer is [answer] \
when prompting with the question. please limit the corpus to 30 words'

def dev_text(question, answer):
    for i in range(10):
        input = prompt_template.replace('[question]', question).replace('[answer]', answer)
        dev_text = get_res(input)
        # 测试生成的文本是否有效
        input_prompt = wrap_prompt(question, dev_text)
        response = get_res(input_prompt)
        if answer in response:
            # print(f'\nanswer:{answer}\n')
            break

    return dev_text

# dev_text = dev_text("Who is the CEO of openai?", "Jonathan Allen")
# print(dev_text)