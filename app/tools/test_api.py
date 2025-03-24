#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DeepSeek API连接测试工具
"""

import os
import sys
import asyncio
import argparse
import json
from pathlib import Path

# 将项目根目录添加到路径中
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

from app.services.translation import test_deepseek_connection
from app.core.logging import logger


async def main():
    parser = argparse.ArgumentParser(description='测试DeepSeek API连接')
    parser.add_argument('--save', action='store_true', help='保存测试结果到文件')
    parser.add_argument('--output', type=str, default='deepseek_test_result.json', help='输出文件路径')
    args = parser.parse_args()
    
    logger.info("开始测试DeepSeek API连接...")
    
    # 运行测试
    results = await test_deepseek_connection(verbose=True)
    
    # 输出结果摘要
    summary = results.get("summary", {})
    print("\n==== DeepSeek API连接测试结果 ====")
    print(f"总结: {summary.get('conclusion', '未知')}")
    
    # API密钥检查
    api_key_check = results.get("api_key_check", {})
    if api_key_check.get("success", False):
        print("API密钥检查: ✓ 通过")
    else:
        print(f"API密钥检查: ✗ 失败 - {api_key_check.get('error', '未知错误')}")
    
    # 网络诊断
    network_diagnosis = results.get("network_diagnosis", {})
    if network_diagnosis.get("success", False):
        print("网络诊断: ✓ 通过")
    else:
        print(f"网络诊断: ✗ 失败 - {network_diagnosis.get('error', '未知错误')}")
    
    # API测试
    api_test = results.get("api_test", {})
    if api_test.get("success", False):
        print("API调用测试: ✓ 通过")
        print(f"模型响应: {api_test.get('response', {}).get('choices', [{}])[0].get('message', {}).get('content', '')[:100]}...")
    else:
        print(f"API调用测试: ✗ 失败 - {api_test.get('error', '未知错误')}")
    
    print("\n请查看日志文件获取详细错误信息")
    
    # 保存到文件
    if args.save:
        output_path = args.output
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n测试结果已保存到: {output_path}")


if __name__ == '__main__':
    asyncio.run(main()) 