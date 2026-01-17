import langextract as lx
from langextract.providers import ollama
import json
import textwrap
import traceback

# ===================== 1. 构造长文本 =====================
long_text = """
2025全品类产品价目表
【数码类】
智能手表：基础款1299元，顶配款1999元；无线耳机：标准版399元，降噪款599元；
平板电脑：11英寸2999元，13英寸4999元；笔记本电脑：轻薄本4999元，游戏本8999元；
【家居类】
智能台灯：89元，空气净化器：899元，扫地机器人：1999元；
【海外专区】
电动牙刷：海外版49美元，国内版299元；保温杯：海外版29欧元，国内版199元；
【数码类补充】
移动电源：10000mAh版99元，20000mAh版159元；数据线：Type-C款29元，Lightning款39元；
智能音箱：mini款199元，Pro款399元；投影仪：入门款1299元，高清款2999元；
【家居类补充】
电热毯：单人款89元，双人款129元；加湿器：基础款129元，除菌款299元；
"""

# ===================== 2. 强化格式约束的提取规则 =====================
prompt = textwrap.dedent("""
严格按照以下要求提取文本中所有「产品」和「价格」信息，输出必须是标准JSON格式（可直接解析）：
1. 提取规则：
   - 按类别分组提取，不遗漏任何条目；
   - 为每个条目补充「品类」属性（数码类/家居类/海外专区）；
   - 价格需包含完整数值+单位，货币类型需明确（人民币/美元/欧元）。
2. 格式约束（关键：强制extraction_text为纯字符串）：
   - 输出仅包含JSON结构，无多余文字、注释或换行；
   - JSON根节点为{"extractions": [...]}，每个元素必须严格符合：
     {
       "extraction_class": "产品" | "价格",  // 仅二选一
       "extraction_text": "纯字符串，无嵌套结构",  // 例如"智能手表基础款"、"1299元"
       "attributes": {
         "品类": "数码类|家居类|海外专区",
         "货币类型": "人民币|美元|欧元"
       }
     }
3. 禁止行为：
   - extraction_text 不能是字典、列表等非字符串类型；
   - 不能省略字段、添加多余字段；
   - 不能有非JSON内容（如注释、说明文字）。
4. 正确示例：
{
  "extractions": [
    {"extraction_class":"产品","extraction_text":"智能手表基础款","attributes":{"品类":"数码类","货币类型":"人民币"}},
    {"extraction_class":"价格","extraction_text":"1299元","attributes":{"品类":"数码类","货币类型":"人民币"}}
  ]
}
""")

# 优化示例：完全匹配格式要求
example = lx.data.ExampleData(
    text="数码类：无线鼠标99元；家居类：保温杯199元；海外专区：耳机49美元",
    extractions=[
        lx.data.Extraction(
            extraction_class="产品",
            extraction_text="无线鼠标",  # 纯字符串，无嵌套
            attributes={"品类": "数码类", "货币类型": "人民币"}
        ),
        lx.data.Extraction(
            extraction_class="价格",
            extraction_text="99元",  # 纯字符串，无嵌套
            attributes={"品类": "数码类", "货币类型": "人民币"}
        ),
        lx.data.Extraction(
            extraction_class="产品",
            extraction_text="保温杯",
            attributes={"品类": "家居类", "货币类型": "人民币"}
        ),
        lx.data.Extraction(
            extraction_class="价格",
            extraction_text="199元",
            attributes={"品类": "家居类", "货币类型": "人民币"}
        ),
        lx.data.Extraction(
            extraction_class="产品",
            extraction_text="耳机",
            attributes={"品类": "海外专区", "货币类型": "美元"}
        ),
        lx.data.Extraction(
            extraction_class="价格",
            extraction_text="49美元",
            attributes={"品类": "海外专区", "货币类型": "美元"}
        )
    ]
)

# ===================== 3. 修正后的提取配置（核心优化） =====================
def main():
    try:
        # 1. 显式初始化Ollama模型，重点优化超时和性能参数
        ollama_model = ollama.OllamaLanguageModel(
            model_id="gemma2:2b",
            model_url="http://localhost:11434",
            format_type=lx.core.types.FormatType.JSON,  # 强制JSON输出
            timeout=600,  # 核心：延长模型超时时间至10分钟
            # 模型性能优化参数
            **{
                "temperature": 0.0,       # 0温度完全确定性输出
                "max_output_tokens": 8000, # 增大输出token上限
                "num_ctx": 8192,           # 提升上下文窗口（适配gemma2:2b的最大支持）
                "num_threads": 8,          # 增加推理线程数（根据CPU核心数调整，如8/16）
                "keep_alive": 600          # 延长连接保持时间
            }
        )

        # 2. 执行提取，优化长文档处理参数
        result = lx.extract(
            text_or_documents=long_text,
            prompt_description=prompt,
            examples=[example],
            model=ollama_model,  # 直接传入模型实例
            # 格式解析配置
            resolver_params={
                "format_handler": ollama.OLLAMA_FORMAT_HANDLER,
                "suppress_parse_errors": False,  # 关闭容错，强制正确格式
                "enable_fuzzy_alignment": True,
                "fuzzy_alignment_threshold": 0.9
            },
            # 长文档优化：增大分块粒度，减少推理次数
            extraction_passes=1,
            max_workers=1,
            max_char_buffer=2000,       # 核心：增大分块粒度，减少模型调用次数
            show_progress=True,
            # 补充模型参数
            language_model_params={
                "timeout": 600,          # 与模型超时保持一致
                "keep_alive": 600
            }
        )

        # ===================== 4. 结果分析 =====================
        category_stats = {}
        for ext in result.extractions:
            # 安全获取属性
            category = ext.attributes.get("品类", "未知")
            currency = ext.attributes.get("货币类型", "未知")
            if category not in category_stats:
                category_stats[category] = {
                    "产品": [],
                    "价格": [],
                    "货币类型": set()
                }
            if ext.extraction_class == "产品":
                category_stats[category]["产品"].append(ext.extraction_text)
            elif ext.extraction_class == "价格":
                category_stats[category]["价格"].append(ext.extraction_text)
            category_stats[category]["货币类型"].add(currency)

        print("=== 长文档提取结果统计 ===")
        for category, data in category_stats.items():
            print(f"\n【{category}】")
            print(f"货币类型：{', '.join(data['货币类型'])}")
            print(f"提取产品数：{len(data['产品'])}，产品列表：{data['产品']}")
            print(f"提取价格数：{len(data['价格'])}，价格列表：{data['价格']}")

        # ===================== 5. 生成可视化文件 =====================
        result_iterator = iter([result])
        lx.io.save_annotated_documents(
            result_iterator,
            output_name="long_doc_extraction.jsonl",
            output_dir="."
        )
        html_content = lx.visualize("long_doc_extraction.jsonl")
        with open("long_doc_extraction_visual.html", "w", encoding="utf-8") as f:
            f.write(html_content.data if hasattr(html_content, 'data') else html_content)
        print("\n✅ 长文档可视化文件已生成：long_doc_extraction_visual.html")

    except Exception as e:
        print(f"\n❌ 执行出错：{type(e).__name__} - {str(e)}")
        traceback.print_exc()

        # 额外排查：检查Ollama服务是否正常
        try:
            import requests
            ollama_health = requests.get("http://localhost:11434/api/tags", timeout=10)
            print(f"\n🔍 Ollama服务状态：{ollama_health.status_code}")
            print(f"🔍 已加载模型：{ollama_health.json()}")
        except Exception as health_e:
            print(f"\n❌ Ollama服务未正常运行：{str(health_e)}")
            print("💡 建议：执行 `ollama serve` 启动服务，或 `ollama pull gemma2:2b` 确认模型已下载")

if __name__ == "__main__":
    # 前置检查：确保Ollama模型已下载并启动
    print("🔍 检查Ollama环境...")
    main()