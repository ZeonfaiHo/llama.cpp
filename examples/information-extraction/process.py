# 完整代码如下：

import pandas as pd
import subprocess

def find_lines_after_title(input_string, title_prefix, item_prefix):
    lines = input_string.split('\n')  # 将字符串按行分割
    result_lines = []
    collect = False  # 开始时设置为不收集行

    for line in lines:
        # 检查当前行是否是我们要查找的标题行
        if line.startswith(title_prefix):
            collect = True  # 找到标题行，开始收集
            continue  # 跳过标题行
        
        # 如果当前正在收集，检查行的开始是否为给定前缀
        if collect and line.startswith(item_prefix):
            result_lines.append(line)
        elif collect:
            # 如果当前行不以给定前缀开头，则停止收集
            break

    if len(result_lines) != 0:
        return result_lines
    else:
        return ['',]


def extract_time(text):

    # 打开task-people文件并读取内容
    with open('task-time.txt', 'r', encoding='utf-8') as file:
        content = file.read()

    # 替换<<news>>为变量text中的内容
    content = content.replace('<<news>>', text)

    # 将替换后的内容写入input.txt中
    with open('input.txt', 'w', encoding='utf-8') as file:
        file.write(content)

    command = "../../build/bin/main -m ../../models/mistral-7b-instruct-v0.1.Q5_K_M.gguf -f ./input.txt --n-gpu-layers 40 -n 32 --temp 0"

    process = subprocess.run(command, 
                            shell=True, text=True, capture_output=True, errors='ignore')

    text = process.stdout

    res = find_lines_after_title(text, '时间：', '- ')
    res = res[0][2:]
    return res


def extract_person(text):

    # 打开task-people文件并读取内容
    with open('task-people.txt', 'r', encoding='utf-8') as file:
        content = file.read()

    # 替换<<news>>为变量text中的内容
    content = content.replace('<<news>>', text)

    # 将替换后的内容写入input.txt中
    with open('input.txt', 'w', encoding='utf-8') as file:
        file.write(content)

    command = "../../build/bin/main -m ../../models/mistral-7b-instruct-v0.1.Q5_K_M.gguf -f ./input.txt --n-gpu-layers 40 -n 128 --temp 0"

    process = subprocess.run(command, 
                            shell=True, text=True, capture_output=True, errors='ignore')

    text = process.stdout

    res = find_lines_after_title(text, '人物：', '- ')
    if len(res[0]) > 2:
        res = [line[2:] for line in res]
        return str(res)
    else:
        return ''

def extract_place(text):

    # 打开task-people文件并读取内容
    with open('task-place.txt', 'r', encoding='utf-8') as file:
        content = file.read()

    # 替换<<news>>为变量text中的内容
    content = content.replace('<<news>>', text)

    # 将替换后的内容写入input.txt中
    with open('input.txt', 'w', encoding='utf-8') as file:
        file.write(content)

    command = "../../build/bin/main -m ../../models/mistral-7b-instruct-v0.1.Q5_K_M.gguf -f ./input.txt --n-gpu-layers 40 -n 256 --temp 0"

    process = subprocess.run(command, 
                            shell=True, text=True, capture_output=True, errors='ignore')

    text = process.stdout

    res = find_lines_after_title(text, '地点：', '- ')
    res = [line[2:] for line in res]
    return str(res)


def extract_speech(text):

    # 打开task-people文件并读取内容
    with open('task-speech.txt', 'r', encoding='utf-8') as file:
        content = file.read()

    # 替换<<news>>为变量text中的内容
    content = content.replace('<<news>>', text)

    # 将替换后的内容写入input.txt中
    with open('input.txt', 'w', encoding='utf-8') as file:
        file.write(content)

    command = "../../build/bin/main -m ../../models/mistral-7b-instruct-v0.1.Q5_K_M.gguf -f ./input.txt --n-gpu-layers 40 -n 512 --temp 0"

    process = subprocess.run(command, 
                            shell=True, text=True, capture_output=True, errors='ignore')

    text = process.stdout

    res = find_lines_after_title(text, '言论：', '- ')
    res = [line[2:] for line in res]
    return str(res)


# 定义一个空的处理函数，用户稍后会自行定义
def process_text(text):
    # 用户将定义如何从文本中提取时间、地点、人物和言论
    time, place, person, speech = None, None, None, None
    # 这里应该是用户定义的文本处理逻辑

    time = extract_time(text)
    place = extract_place(text)
    person = extract_person(text)
    speech = extract_speech(text)

    print('Result: ' + str((time, place, person, speech)))    
    return time, place, person, speech

# df = pd.read_csv('./processed_news.csv')
df = pd.read_csv('./extract_information.csv.backup')

# 给DataFrame添加缺失的列
if '时间' not in df.columns:
    df['时间'] = None
if '地点' not in df.columns:
    df['地点'] = None
if '人物' not in df.columns:
    df['人物'] = None
if '言论' not in df.columns:
    df['言论'] = None

# 对第5列数据进行处理，并填入新列
# 假设需要处理的文本在'文本'列中
for index, row in df.iterrows():
    if pd.notna(row['时间']) and pd.notna(row['地点']) and pd.notna(row['人物']) and pd.notna(row['言论']):
        continue

    time, place, person, speech = process_text(row['文本'])
    df.at[index, '时间'] = time
    df.at[index, '地点'] = place
    df.at[index, '人物'] = person
    df.at[index, '言论'] = speech

    df.to_csv('./extract_information.csv', index=False)

