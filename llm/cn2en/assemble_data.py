# 装配语料

import json
import re

zh_file = 'source/chinese1.txt'
en_file = 'source/english1.txt'

output_file = 'translation_dataset.jsonl'

print('⚙️ 开始组装语料库...')

# with 上下文管理器， 自动关闭。
# 同时打开多个文件， open 打开文件， 指定读写方式和编码

with (
    open(zh_file, 'r', encoding='utf-8') as f_zh,
    open(en_file, 'r', encoding='utf-8') as f_en,
    open(output_file, 'w', encoding='utf-8') as f_output
):
    count = 0
    # zip 打包函数， 想多个文件一行一行对其
    # zip([1,2,3], ['a','b','c'])  => (1, 'a'), (2, 'b'), (3, 'c')
    for zh_line, en_line in zip(f_zh, f_en):
        # 去除两边空格 \n tab  类似 js trim
        zh_text = zh_line.strip()

        # re 正则表达式库   r'[\u200b-\u200f\ufeff]'  r'' 代表匹配原始字符串，因为 python 默认会将 "\u200b" 当做 unicode 转义字符

        # 这段代码是为了去除 一些看不见的 Unicode垃圾字符
        zh_text = re.sub(r'[\u200b-\u200f\ufeff]', '', zh_text)

        en_text = en_line.strip()
        en_text = re.sub(r'[\u200b-\u200f\ufeff]', '', en_text)


        if zh_text and en_text:
            item = {
                'zh': zh_text,
                'en': en_text
            }
            # 写入一行 (JSONL 要求每一行是一个完整的 JSON)
            f_output.write(json.dumps(item, ensure_ascii=False) + '\n')

        count +=1

        if count % 1000 == 0:
            print(f"已处理 {count} 行...")

print(f"✅ 完成！共组装 {count} 条双语数据，已存入 {output_file}")



