import langextract as lx
from langextract.providers import ollama
import warnings

# 忽略示例对齐的非关键警告（可选）
warnings.filterwarnings("ignore", category=UserWarning)

# ===================== 1. 医疗文本提取规则（优化prompt，适配LLM理解） =====================
prompt = """
严格从临床病历中提取以下结构化信息，要求：
1. 提取类别：药品名称、剂量、给药途径、用药频率；
2. 每个提取条目必须通过「药品组」属性关联到对应的药品（药品组值为药品名称的核心词）；
3. 若某类别信息缺失，标注为「无」；
4. 完全复用原文中的医学术语，不修改格式（如10U、500mg保持原样）。
"""

# ===================== 2. 优化Few-shot示例（提升对齐准确性） =====================
example = lx.data.ExampleData(
    text="患者因高血压口服硝苯地平缓释片，每次20mg，每日2次，连续服用14天。",
    extractions=[
        lx.data.Extraction(
            extraction_class="药品名称",
            extraction_text="硝苯地平缓释片",
            attributes={"药品组": "硝苯地平缓释片"}  # 药品组与名称一致，减少对齐误差
        ),
        lx.data.Extraction(
            extraction_class="剂量",
            extraction_text="20mg/次",
            attributes={"药品组": "硝苯地平缓释片"}
        ),
        lx.data.Extraction(
            extraction_class="给药途径",
            extraction_text="口服",
            attributes={"药品组": "硝苯地平缓释片"}
        ),
        lx.data.Extraction(
            extraction_class="用药频率",
            extraction_text="每日2次",
            attributes={"药品组": "硝苯地平缓释片"}
        )
    ]
)

# ===================== 3. 待提取的病历文本 =====================
medical_text = """
【病历摘要】
患者女性，65岁，2型糖尿病病史5年，本次因血糖控制不佳入院：
1. 胰岛素注射液（短效）：皮下注射，每次10U，每餐前30分钟，持续使用；
2. 二甲双胍片：口服，每次500mg，每日3次，连服30天；
3. 阿托伐他汀钙片：口服，每次20mg，每晚1次，长期服用。
"""

# ===================== 4. 初始化Ollama模型（优化超时和性能参数） =====================
ollama_model = ollama.OllamaLanguageModel(
    model_id="qwen3:8b",
    model_url="http://localhost:11434",
    timeout=300,  # 延长超时时间（默认120s→300s）
    temperature=0.1,  # 降低随机性，提升医疗文本提取准确性
    num_ctx=4096,  # 增大上下文窗口，适配长病历
    keep_alive=10*60  # 延长连接保持时间
)

# ===================== 5. 执行提取（优化参数） =====================
try:
    result = lx.extract(
        text_or_documents=medical_text,
        prompt_description=prompt,
        examples=[example],
        model=ollama_model,  # 直接传入模型实例，避免模型ID解析冲突
        resolver_params={"format_handler": ollama.OLLAMA_FORMAT_HANDLER},
        show_progress=True,
        language_model_params={
            "timeout": 300,  # 再次确认超时参数
            "num_threads": 8  # 利用多线程加速（根据CPU核心数调整）
        }
    )

    # ===================== 6. 结构化输出 =====================
    drug_groups = {}
    for ext in result.extractions:
        drug_group = ext.attributes.get("药品组", "未知")
        if drug_group not in drug_groups:
            drug_groups[drug_group] = {}
        drug_groups[drug_group][ext.extraction_class] = ext.extraction_text

    print("=== 医疗文本结构化提取结果 ===")
    for drug, info in drug_groups.items():
        print(f"\n【药品：{drug}】")
        print(f"药品名称：{info.get('药品名称', '无')}")
        print(f"剂量：{info.get('剂量', '无')}")
        print(f"给药途径：{info.get('给药途径', '无')}")
        print(f"用药频率：{info.get('用药频率', '无')}")

    # ===================== 7. 生成可视化文件 =====================
    result_iterator = iter([result])
    lx.io.save_annotated_documents(result_iterator, output_name="medical_extraction_01.jsonl", output_dir=".")
    html_content = lx.visualize("medical_extraction_01.jsonl")
    with open("medical_extraction_visual_01.html", "w", encoding="utf-8") as f:
        # 兼容不同环境的html_content格式
        f.write(html_content.data if hasattr(html_content, 'data') else html_content)
    print("\n✅ 医疗文本可视化文件已生成：medical_extraction_visual_01.html")

except Exception as e:
    print(f"\n❌ 执行出错：{type(e).__name__}: {str(e)}")
    # 调试信息：检查Ollama服务状态
    import subprocess
    try:
        subprocess.run(["ollama", "ps"], check=True)
    except subprocess.CalledProcessError:
        print("⚠️ Ollama服务未启动，请先执行：ollama serve")