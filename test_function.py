import asyncio
import json
import os
from app.services.translation import get_video_context_from_llm
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

async def test_get_video_context():
    """
    测试get_video_context_from_llm函数
    """
    # 测试视频标题和频道名
    title = "Stanford CS25: V5 I Overview of Transformers"
    channel_name = "Stanford Online"
    
    try:
        # 调用函数获取视频上下文
        content = await get_video_context_from_llm(title, channel_name)
        
        # 将JSON字符串转换为Python对象
        context_data = json.loads(content)
        
        # 打印结果
        print("原始响应:")
        print(content)
        
        print("\n处理后的数据:")
        print(json.dumps(context_data, ensure_ascii=False, indent=2))
        
        return context_data
    except Exception as e:
        print(f"测试失败: {str(e)}")
        raise

import json

def process_json_data(json_data):
    # 将JSON字符串转换为Python字典（如果输入是字符串的话）
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data
    
    # 任务1: 提取can_judge为true的字段，转为纯文本
    text_output = []
    
    # 遍历每个step
    for step_key, step_value in data.items():
        # 检查是否包含can_judge且为true
        if step_value.get("can_judge", False) == True:
            # 提取不同类型的字段
            if "channel_info" in step_value:
                text_output.append(step_value["channel_info"])
            if "content_inference" in step_value:
                text_output.append(step_value["content_inference"])
            if "translation_strategies" in step_value and isinstance(step_value["translation_strategies"], list):
                for strategy in step_value["translation_strategies"]:
                    text_output.append(strategy)
    
    # 将文本列表转换为换行分隔的字符串
    formatted_text = "\n".join(text_output)
    
    # 任务2: 提取step3中的translation_strategies
    translation_strategies = {}
    if "step3" in data and "translation_strategies" in data["step3"]:
        translation_strategies = {
            "translation_strategies": data["step3"]["translation_strategies"]
        }
    
    # 直接返回两个独立的变量，而不是字典
    return formatted_text, translation_strategies

# 运行测试函数
if __name__ == "__main__":
    data = asyncio.run(test_get_video_context())

    # 处理数据 - 现在返回两个独立变量
    text_content, strategies = process_json_data(data)
    
    # 分别使用这两个变量
    print("提取的纯文本（存储在变量 text_content 中）：")
    print(text_content)
    print("\n提取的translation_strategies（存储在变量 strategies 中）：")
    print(json.dumps(strategies, ensure_ascii=False, indent=4))