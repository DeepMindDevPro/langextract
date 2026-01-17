import langextract as lx
from langextract.providers import ollama

# ===================== 1. 定义提取任务（核心：Prompt + Few-shot 示例） =====================
# 提取规则描述：明确要提取的内容、格式、约束
prompt = """
提取文本中的「产品」和「价格」信息，要求：
1. 按文本中出现顺序提取，不重复、不释义；
2. 为每个产品补充「货币类型」属性（从文本中识别）；
3. 价格需包含完整数值+单位。
"""

# Few-shot 示例：给模型参考的「输入-输出」范例（决定提取效果的关键）
example = lx.data.ExampleData(
    text="2025新款笔记本电脑售价5999元，无线鼠标99元",  # 示例输入文本
    extractions=[  # 示例提取结果（定义结构化格式）
        lx.data.Extraction(
            extraction_class="产品",  # 提取类别（自定义）
            extraction_text="笔记本电脑",  # 提取的文本内容
            attributes={"货币类型": "人民币"}  # 自定义属性
        ),
        lx.data.Extraction(
            extraction_class="价格",
            extraction_text="5999元",
            attributes={"货币类型": "人民币"}
        ),
        lx.data.Extraction(
            extraction_class="产品",
            extraction_text="无线鼠标",
            attributes={"货币类型": "人民币"}
        ),
        lx.data.Extraction(
            extraction_class="价格",
            extraction_text="99元",
            attributes={"货币类型": "人民币"}
        )
    ]
)

# ===================== 2. 待提取的输入文本 =====================
input_text = """
2025数码新品发布会：
- 智能手表售价1299元，运动手环299元；
- 海外版耳机定价199美元，国内版1299元。
"""

# ===================== 3. 调用 LangExtract 提取（指定本地 Ollama 模型） =====================
result = lx.extract(
    text_or_documents=input_text,
    prompt_description=prompt,
    examples=[example],  # 传入Few-shot示例
    model_id="gemma2:2b",  # 本地Ollama模型ID
    model_url="http://localhost:11434",  # Ollama默认地址
    resolver_params={"format_handler": ollama.OLLAMA_FORMAT_HANDLER},  # 适配Ollama的格式处理器
    show_progress=True  # 显示提取进度
)

# ===================== 4. 解析提取结果（核心：结构化+原文溯源） =====================
print("=== 提取结果汇总 ===")
print(f"总提取条目数：{len(result.extractions)}")
print("\n=== 逐条解析结果 ===")
for idx, ext in enumerate(result.extractions):
    print(f"\n【条目{idx+1}】")
    print(f"提取类别：{ext.extraction_class}")
    print(f"提取文本：{ext.extraction_text}")
    print(f"自定义属性：{ext.attributes}")
    # 原文溯源：提取内容在原文中的字符位置（LangExtract核心特性）
    if ext.char_interval:
        start = ext.char_interval.start_pos
        end = ext.char_interval.end_pos
        print(f"原文位置：第{start}-{end}字符")
        print(f"原文对应内容：{input_text[start:end]}")

# ===================== 5. 可视化结果（直观验证提取准确性） =====================
# 保存结果到JSONL文件
result_iterator = iter([result])
lx.io.save_annotated_documents(result_iterator, output_name="basic_extraction_01.jsonl", output_dir=".")
# 生成交互式HTML可视化文件
html_content = lx.visualize("basic_extraction_01.jsonl")
with open("basic_extraction_visual_01.html", "w", encoding="utf-8") as f:
    f.write(html_content)
print("\n✅ 可视化文件已生成：basic_extraction_visual_01.html（打开可查看原文高亮的提取结果）")