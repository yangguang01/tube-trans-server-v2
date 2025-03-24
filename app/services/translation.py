import os
import json
import asyncio
import re
import yt_dlp
import replicate
from datetime import datetime
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import AsyncOpenAI
import aiohttp
from functools import wraps
import openai
import httpx

from app.core.config import REPLICATE_API_TOKEN, DEEPSEEK_API_KEY, RETRY_ATTEMPTS, BATCH_SIZE, MAX_CONCURRENT_TASKS, API_TIMEOUT
from app.core.logging import logger
from app.utils.file_utils import cleanup_audio_file


def download_audio_webm(url, file_path):
    """
    从指定 URL 下载音频（仅下载 webm 格式的音频流）
    
    参数:
        url (str): 媒体资源的 URL
        file_path (Path): 保存音频的路径
        
    返回:
        Path: 下载后的音频文件路径
    """
    try:
        logger.info(f"开始下载视频: {url}")
        
        ydl_opts = {
            'format': 'bestaudio[ext=webm]',
            'outtmpl': str(file_path),
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            },
            'force_ipv4': True
            #'proxy': 'socks5://8t4v58911-region-US-sid-JaboGcGm-t-5:wl34yfx7@us2.cliproxy.io:443',

        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            
        logger.info(f"视频下载完成: {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"下载失败: {str(e)}", exc_info=True)
        raise


def get_video_info(url):
    """
    获取YouTube视频信息
    
    参数:
        url (str): YouTube URL
        
    返回:
        dict: 视频信息字典
    """
    try:
        logger.info(f"获取视频信息: {url}")
        
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'skip_download': True,
            'forcejson': True,
            #'proxy': 'socks5://8t4v58911-region-US-sid-JaboGcGm-t-5:wl34yfx7@us2.cliproxy.io:443',

        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            
        logger.info(f"获取视频信息成功: {info.get('title', 'Unknown')}")
        return info
    except Exception as e:
        logger.error(f"获取视频信息失败: {str(e)}", exc_info=True)
        raise


def transcribe_audio(file_path):
    """
    使用 Replicate 的 Whisper 模型将音频文件转换为文本
    
    参数:
        file_path (Path): 音频文件路径
        
    返回:
        dict: 转写结果
    """
    try:
        logger.info(f"开始转写音频: {file_path}")
        
        # 设置环境变量
        os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN
        
        # 设置超长的超时时间 - 音频转写需要很长时间
        AUDIO_TIMEOUT = 3600  # 一小时超时
        
        # 直接修改httpx全局默认超时 - 这是最直接的方法
        httpx._config.DEFAULT_TIMEOUT_CONFIG = httpx.Timeout(
            connect=1200.0,             # 连接超时
            read=AUDIO_TIMEOUT,       # 读取超时 - 关键参数
            write=1200.0,              # 写入超时
            pool=1200.0                 # 连接池超时
        )
        
        logger.info(f"已设置全局HTTPX超时为: 连接={60}秒, 读取={AUDIO_TIMEOUT}秒")
        
        logger.info("正在调用Replicate API进行音频转写，可能需要较长时间...")
        
        # 使用Replicate的Client，而不是直接使用run函数
        from replicate.client import Client
        
        # 创建客户端实例 - 使用自定义的超长超时
        client = Client(
            api_token=REPLICATE_API_TOKEN,
            timeout=AUDIO_TIMEOUT  # 直接设置超时为一小时
        )
        
        # 创建自定义的HTTPX客户端，替换默认客户端
        custom_httpx_client = httpx.Client(timeout=AUDIO_TIMEOUT)
        
        # 确保使用我们的客户端
        # client._client = custom_httpx_client  # 这一行可能不起作用，取决于replicate库的实现
        
        with open(file_path, "rb") as audio_file:
            input_data = {
                "file": audio_file,
                "prompt": "",
                "language": "en",
                "num_speakers": 2
            }
            
            logger.info("开始上传音频文件并等待转写结果...")
            
            # 使用自定义客户端运行模型
            output = client.run(
                "thomasmol/whisper-diarization:d8bc5908738ebd84a9bb7d77d94b9c5e5a3d867886791d7171ddb60455b4c6af",
                input=input_data
            )
            
        logger.info("音频转写完成")
        return output
    except Exception as e:
        logger.error(f"转写失败: {str(e)}", exc_info=True)
        raise


def format_time(seconds):
    """将秒数转换为 SRT 格式的时间字符串，格式为 hh:mm:ss,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"


def json_to_srt(data):
    """从 JSON 数据中提取 segments 字段，转换为 SRT 格式的文本"""
    srt_lines = []
    for idx, segment in enumerate(data.get("segments", []), start=1):
        start_time = format_time(segment["start"])
        end_time = format_time(segment["end"])
        text = segment["text"]
        srt_lines.append(str(idx))
        srt_lines.append(f"{start_time} --> {end_time}")
        srt_lines.append(text)
        srt_lines.append("")  # 添加空行分隔不同字幕段
    return "\n".join(srt_lines)


def merge_incomplete_sentences(subtitles):
    """将英文字幕中的内容合并为完整句子"""
    # 按行分割字幕文本
    lines = [line.strip() for line in subtitles.split('\n') if line.strip()]

    # 存储合并后的句子
    merged_sentences = []
    current_sentence = ''

    for line in lines:
        if not line.isdigit() and '-->' not in line and line.strip() != '':
            # 添加当前行到当前句子
            current_sentence += ' ' + line if current_sentence else line

            # 检查是否为完整句子
            if any(current_sentence.endswith(symbol) for symbol in ['.', '?', '!']):
                merged_sentences.append(current_sentence)
                current_sentence = ''

    # 确保最后一句也被添加（如果它是完整的）
    if current_sentence:
        merged_sentences.append(current_sentence)

    # 将每个句子转换为字典，并添加序号
    numbered_and_sentences = {i: sentence for i, sentence in enumerate(merged_sentences, start=1)}

    return numbered_and_sentences


# 通用的异步重试装饰器
def async_retry(max_attempts=None, exceptions=None):
    """异步函数的重试装饰器"""
    if max_attempts is None:
        max_attempts = RETRY_ATTEMPTS
    if exceptions is None:
        exceptions = (Exception,)
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    # 指数退避策略
                    wait_time = min(1 * (2 ** attempt), 8)
                    logger.warning(f"尝试 {attempt+1}/{max_attempts} 失败: {str(e)}，等待 {wait_time}秒后重试")
                    await asyncio.sleep(wait_time)
            # 所有重试都失败了
            raise last_exception or Exception("最大重试次数已用尽")
        return wrapper
    return decorator


@async_retry()
async def safe_api_call_async(client, messages, model):
    """安全的异步API调用，内置重试机制"""
    try:
        # 使用自定义超时设置 - 使用浮点数表示秒数，而不是ClientTimeout对象
        timeout_seconds = API_TIMEOUT  # 直接使用秒数
        
        # 创建新的AsyncOpenAI客户端实例，包含超时设置
        temp_client = AsyncOpenAI(
            api_key=client.api_key,
            base_url=client.base_url,  # 使用传入客户端的base_url
            timeout=timeout_seconds  # 使用秒数作为timeout，而不是ClientTimeout对象
        )
        
        logger.info(f"开始调用DeepSeek API, 模型:{model}, base_url:{client.base_url}, timeout:{timeout_seconds}秒")
        
        # 使用新客户端发送请求
        response = await temp_client.chat.completions.create(
            model=model,
            response_format={'type': "json_object"},
            messages=messages,
            temperature=0.3,
            top_p=0.7,
            frequency_penalty=0,
            presence_penalty=0,
        )

        # 检查响应结构
        if not hasattr(response, 'choices') or len(response.choices) == 0:
            logger.error(f"无效的API响应结构: {response}")
            raise ValueError("无效的API响应结构")

        message = response.choices[0].message
        if not hasattr(message, 'content'):
            logger.error(f"响应中缺少翻译内容: {message}")
            raise ValueError("响应中缺少翻译内容")

        # 预验证JSON格式
        try:
            json_content = json.loads(message.content)
            logger.debug(f"API调用成功返回有效JSON")
        except json.JSONDecodeError as e:
            logger.error(f"JSON预验证失败: {message.content}")
            raise

        return response

    except openai.APIConnectionError as e:
        # 记录连接错误详情
        import traceback
        
        # 获取错误代码和HTTP状态码
        status_code = getattr(e, 'status_code', 'unknown')
        error_code = getattr(e, 'code', 'unknown')
        
        # 获取底层异常详情
        cause = e.__cause__ if hasattr(e, '__cause__') else None
        cause_type = type(cause).__name__ if cause else 'None'
        cause_str = str(cause) if cause else 'None'
        
        # 输出详细错误信息
        logger.error(f"DeepSeek API连接错误详情: {str(e)}")
        logger.error(f"状态码: {status_code}, 错误码: {error_code}")
        logger.error(f"底层异常: {cause_type}: {cause_str}")
        logger.error(f"堆栈跟踪: {traceback.format_exc()}")
        
        # 重新抛出异常
        raise
        
    except openai.APITimeoutError as e:
        logger.error(f"DeepSeek API超时: {str(e)}")
        logger.error(f"超时详情: {traceback.format_exc()}")
        raise
        
    except openai.RateLimitError as e:
        # 记录限流错误详情
        status_code = getattr(e, 'status_code', 'unknown')
        error_code = getattr(e, 'code', 'unknown')
        
        logger.error(f"DeepSeek API速率限制: {str(e)}")
        logger.error(f"状态码: {status_code}, 错误码: {error_code}")
        raise
        
    except openai.APIResponseValidationError as e:
        # 记录响应验证错误详情
        status_code = getattr(e, 'status_code', 'unknown')
        error_code = getattr(e, 'code', 'unknown')
        
        logger.error(f"DeepSeek API响应验证错误: {str(e)}")
        logger.error(f"状态码: {status_code}, 错误码: {error_code}")
        raise
        
    except openai.AuthenticationError as e:
        # 记录验证错误详情
        status_code = getattr(e, 'status_code', 'unknown')
        error_code = getattr(e, 'code', 'unknown')
        
        logger.error(f"DeepSeek API验证错误: {str(e)}")
        logger.error(f"状态码: {status_code}, 错误码: {error_code}")
        raise
        
    except openai.BadRequestError as e:
        # 记录请求错误详情
        status_code = getattr(e, 'status_code', 'unknown')
        error_code = getattr(e, 'code', 'unknown')
        param = getattr(e, 'param', 'unknown')
        
        logger.error(f"DeepSeek API请求错误: {str(e)}")
        logger.error(f"状态码: {status_code}, 错误码: {error_code}, 参数: {param}")
        raise
        
    except Exception as e:
        # 记录其他异常
        logger.error(f"异步API调用失败: {str(e)}")
        logger.error(f"异常类型: {type(e).__name__}")
        logger.error(f"堆栈跟踪: {traceback.format_exc()}")
        raise


async def process_chunk(chunk, custom_prompt, model, client, semaphore):
    """处理单个翻译批次"""
    async with semaphore:
        result = {
            'translations': {}
        }
        
        chunk_string = ''.join(f"{number}: {sentence}\n" for number, sentence in chunk)
        check_chunk_string = chunk_string.count('\n')
        first_item_number = chunk[0][0] if chunk else "N/A"
        end_item_number = first_item_number + check_chunk_string - 1
        
        # 构造系统提示
        trans_json_user_prompt_v3 = f'''
        # Role
        You are a skilled translator specializing in converting English subtitles into natural and fluent Chinese while maintaining the original meaning.

        # Background information of the translation content
        {custom_prompt}

        ## Skills
        ### Skill 1: Line-by-Line Translation
        - Emphasize the importance of translating English subtitles line by line to ensure accuracy and coherence.
        - Strictly follow the rule of translating each subtitle line individually based on the input content.
        In cases where a sentence is split, such as:
        125.	But what's even more impressive is their U.S.
        126.	commercial revenue projection.

        Step 1: Merge the split sentence into a complete sentence.
        Step 2: Translate the merged sentence into Chinese.
        Step 3: When outputting the Chinese translation, insert the translated result into both of the original split sentence positions.

        Example of this process:
          125.	But what's even more impressive is their U.S.
        但更令人印象深刻的是他们的美国业务
          126.	commercial revenue projection.
        但更令人印象深刻的是他们的美国业务

        ### Skill 2: Contextual Translation
        - Consider the context of the video to ensure accuracy and coherence in the translation.
        - When slang or implicit information appears in the original text, do not translate it literally. Instead, adapt it to align with natural Chinese communication habits.

        ### Skill 3: Handling Complex Sentences
        - Rearrange word order and adjust wording for complex sentence structures to ensure translations are easily understandable and fluent in Chinese.

        ### Skill 4: Proper Nouns and Special Terms
        - Identify proper nouns and special terms enclosed in angle brackets < > within the subtitle text, and retain them in their original English form.

        ### Skill 5: Ignore spelling errors
        - The English content is automatically generated by ASR and may contain spelling errors. Please ignore such errors and translate normally when encountered.

        ## Constraints
        - For punctuation requirements: Do not add a period when the sentence ends
        - The provided subtitles range from line {first_item_number} to line {end_item_number}, totaling {check_chunk_string} lines.
        - Provide the Chinese translation in the specified JSON format:
          ```
          {{
          "1": "<Translation of subtitle line 1>",
          "2": "<Translation of subtitle line 2>",
          "3": "<Translation of subtitle line 3>",
          ...
          }}
          ```
        '''

        try:
            # 初次API调用
            response = await safe_api_call_async(
                client=client,
                messages=[
                    {"role": "system", "content": trans_json_user_prompt_v3},
                    {"role": "user", "content": chunk_string}
                ],
                model=model
            )

            translated_string = response.choices[0].message.content
            trans_to_json = json.loads(translated_string)

            # 行数检查
            check_translated = len(trans_to_json)
            if check_chunk_string == check_translated:
                logger.info(f'编号{first_item_number}一次性通过')
                # 正常处理流程
                new_num_dict = process_transdict_num(trans_to_json, first_item_number, end_item_number)
                translated_dict = process_translated_string(new_num_dict)
                result['translations'].update(translated_dict)
            else:
                # 进入重试逻辑
                retry_prompt_v2 = f'''
                The result of your previous translation attempt was problematic. The content I provided contains {check_chunk_string} lines, but your translation output had {check_translated} lines, indicating a mismatch.

                Please carefully review the translation guidelines and ensure that you translate the subtitles line by line, maintaining a one-to-one correspondence between the original English lines and the translated Chinese lines.

                Translation guidelines:
                - Translate the subtitles line by line, ensuring that the number of lines in your translation matches the number of lines in the original English subtitles.
                - The subtitles to be translated range from line {first_item_number} to line {end_item_number}, totaling {check_chunk_string} lines.
                - When encountering complex sentence structures, rearrange the word order and adjust the wording to ensure the translation is easily understandable and fluent in Chinese while still maintaining the original meaning.
                - Identify proper nouns and special terms enclosed in angle brackets < > within the subtitle text, and keep them in their original English form without translation.
                - Consider the context of the video when translating to ensure accuracy and coherence.

                Video description: {custom_prompt}

                Please provide your revised Chinese translation in the following JSON format:
                {{
                "1": "<Translation of subtitle line 1>",
                "2": "<Translation of subtitle line 2>",
                "3": "<Translation of subtitle line 3>",
                ...
                }}
                '''

                try:
                    # 重试API调用
                    @retry(stop=stop_after_attempt(2))
                    async def retry_call():
                        return await safe_api_call_async(
                            client=client,
                            messages=[
                                {"role": "system", "content": trans_json_user_prompt_v3},
                                {"role": "assistant", "content": translated_string},
                                {"role": "user", "content": retry_prompt_v2}
                            ],
                            model=model
                        )
                        
                    retry_response = await retry_call()
                    
                    retry_translated_string = retry_response.choices[0].message.content

                    # 强制重复验证
                    try:
                        retrytrans_to_json = json.loads(retry_translated_string)
                    except json.JSONDecodeError as e:
                        logger.error(f"重试响应JSON解析失败: {retry_translated_string}")
                        raise

                    # 重复行数检查
                    check_retry = len(retrytrans_to_json)
                    if check_retry == check_chunk_string:
                        # 处理成功重试
                        logger.info("重试有效！")
                        
                        # 对翻译后的字符串进行处理
                        new_num_dict = process_transdict_num(retrytrans_to_json, first_item_number, end_item_number)
                        translated_dict = process_translated_string(new_num_dict)
                        
                        result['translations'].update(translated_dict)
                    else:
                        raise ValueError(f"重试后行数仍不匹配 ({check_retry} vs {check_chunk_string})")

                except Exception as retry_error:
                    logger.error(f"重试失败: {str(retry_error)}")
                    result['translations'][first_item_number] = f"翻译失败: {str(retry_error)}"

        except Exception as main_error:
            logger.error(f"主流程错误: {str(main_error)}")
            result['translations'][first_item_number] = f"关键错误: {str(main_error)}"

        return result


# 处理翻译之后的字符串
def process_translated_string(translated_json):
    # 定义用于匹配中文标点的正则表达式
    chinese_punctuation = r"[\u3000-\u303F\uFF01-\uFFEF<>]"

    # 重新构建带序号的句子格式
    translated_dict = {}

    for number, sentence in translated_json.items():
        # 删除中文标点符号
        sentence = re.sub(chinese_punctuation, ' ', sentence)

        number = int(number)
        # 最后保存成字典
        translated_dict[number] = sentence
    return translated_dict


# 处理翻译之后的字典编号，避免LLM输出的字典编号有误
def process_transdict_num(input_dict, start_num, end_num):
    processed_dict = {}
    for i, (key, value) in enumerate(input_dict.items(), start=start_num):
        new_key = str(i)
        if i <= end_num:
            processed_dict[new_key] = value
        else:
            break
    return processed_dict


# 将原始英文字幕转为字典
def subtitles_to_dict(subtitles):
    """
    Parse subtitles that include a number, a time range, and text.
    Returns a dictionary with numbers as keys and a tuple (time range, text) as values.
    """
    subtitles_dict = {}
    lines = subtitles.strip().split("\n")
    current_number = None
    current_time_range = ""
    current_text = ""

    for line in lines:
        if line.isdigit():
            if current_number is not None:
                subtitles_dict[current_number] = (current_time_range, current_text.strip())
            current_number = int(line)
        elif '-->' in line:
            current_time_range = line
            current_text = ""
        else:
            current_text += line + " "

    subtitles_dict[current_number] = (current_time_range, current_text.strip())

    return subtitles_dict


# 将合并后的英文句子与原始英文字幕做匹配，给合并后的英文添加上时间戳
def map_marged_sentence_to_timeranges(merged_content, subtitles):
    """
    For each merged sentence, find the corresponding subtitles and their time ranges by concatenating
    the subtitles sentences until they match the merged sentence, and merge the time ranges accordingly.
    This version correctly handles multiple merged sentences.
    """
    merged_to_subtitles = {}
    subtitle_index = 0  # Keep track of the current position in the subtitles

    for num, merged_sentence in merged_content.items():
        corresponding_subtitles = []
        start_time = None
        end_time = None
        temp_sentence = ""

        while subtitle_index < len(subtitles):
            sub_num, (time_range, subtitle) = list(subtitles.items())[subtitle_index]
            if start_time is None:
                start_time = time_range.split(' --> ')[0]  # Set the start time of the first subtitle

            temp_sentence += subtitle + " "
            end_time = time_range.split(' --> ')[1]  # Update the end time with each subtitle added
            corresponding_subtitles.append(subtitle)

            # Check if the concatenated subtitles match the merged sentence
            if temp_sentence.strip() == merged_sentence:
                merged_time_range = f"{start_time} --> {end_time}"
                merged_to_subtitles[num] = (merged_time_range, temp_sentence)
                subtitle_index += 1  # Move to the next subtitle for the next iteration
                break

            subtitle_index += 1

    return merged_to_subtitles


# 给中文翻译添加时间轴，生成未经句子长度优化的初始中文字幕
def map_chinese_to_time_ranges(chinese_content, merged_engsentence_to_subtitles):
    chinese_to_time = {}
    chinese_subtitles = []

    # 与句子合并后的英文字幕做匹配
    for num, chinese_sentence in chinese_content.items():
        if num in merged_engsentence_to_subtitles:
            time_ranges, _ = merged_engsentence_to_subtitles[num]
            chinese_to_time[num] = time_ranges, chinese_sentence

            chinese_subtitles.append([
                num,
                time_ranges,
                chinese_sentence
            ])

    return chinese_to_time


def parse_time(time_str):
    """解析时间字符串为datetime对象"""
    return datetime.strptime(time_str, '%H:%M:%S,%f')


def time_to_str(time_obj):
    """将datetime对象转换为时间字符串"""
    return time_obj.strftime('%H:%M:%S,%f')[:-3]


async def translate_with_deepseek_async(numbered_sentences_chunks, custom_prompt, special_terms="", content_name="", model='deepseek-chat'):
    """
    使用DeepSeek异步并行翻译英文字幕到中文
    """
    items = list(numbered_sentences_chunks.items())
    total_translated_dict = {}

    # 处理特殊术语
    if special_terms:
        special_terms = special_terms.rstrip(".")
        special_terms_list = special_terms.split(", ")

    # 创建信号量
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
    
    # 创建异步OpenAI客户端，使用API_TIMEOUT配置超时（使用秒数）
    client = AsyncOpenAI(
        api_key=DEEPSEEK_API_KEY, 
        base_url="https://api.deepseek.com",  # 保持原始URL
        timeout=float(API_TIMEOUT)  # 确保timeout是浮点数
    )
    
    logger.info(f"创建DeepSeek API客户端: base_url=https://api.deepseek.com, timeout={API_TIMEOUT}秒")
    
    # 创建批次处理任务
    tasks = []
    for i in range(0, len(items), BATCH_SIZE):
        chunk = items[i:i + BATCH_SIZE]
        tasks.append(
            process_chunk(chunk, custom_prompt, model, client, semaphore)
        )
    
    # 并行执行所有任务
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 处理结果
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"批次处理失败: {str(result)}")
            continue
        
        # 更新翻译结果
        translations = result.get('translations', {})
        total_translated_dict.update(translations)

    logger.info(f'使用的模型：{model}')

    return total_translated_dict


def format_subtitles(subtitles_dict):
    """将字幕字典格式化为SRT格式字符串"""
    sorted_keys = sorted(subtitles_dict.keys())
    srt_lines = []
    
    for idx, key in enumerate(sorted_keys, start=1):
        time_range, text = subtitles_dict[key]
        srt_lines.append(str(idx))
        srt_lines.append(time_range)
        srt_lines.append(text)
        srt_lines.append("")  # 空行分隔

    return "\n".join(srt_lines)


def robust_transcribe(file_path, max_attempts=3):
    """
    带有重试机制的音频转写函数，处理各种超时和网络错误
    
    参数:
        file_path (Path): 音频文件路径
        max_attempts (int): 最大重试次数
        
    返回:
        dict: 转写结果
    """
    # 定义可以重试的异常类型
    retriable_exceptions = (
        httpx.ReadTimeout, 
        httpx.ConnectTimeout,
        httpx.ReadError,
        httpx.NetworkError,
        ConnectionError,
        TimeoutError
    )
    
    @retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type(retriable_exceptions),
        before_sleep=lambda retry_state: logger.info(
            f"第 {retry_state.attempt_number}/{max_attempts} 次尝试失败，"
            f"等待 {retry_state.next_action.sleep} 秒后重试..."
        )
    )
    def attempt_transcribe():
        return transcribe_audio(file_path)
    
    try:
        logger.info(f"开始转写音频，最多尝试 {max_attempts} 次...")
        return attempt_transcribe()
    except Exception as e:
        logger.error(f"所有转写尝试均失败: {str(e)}", exc_info=True)
        # 重新抛出异常，让调用者处理
        raise


# 修改处理音频接口的调用方式
async def process_audio(audio_path, output_dir, content_name, custom_prompt="", special_terms=""):
    """
    处理音频文件，包括转写和翻译
    
    参数:
        audio_path (Path): 音频文件路径
        output_dir (Path): 输出目录
        content_name (str): 内容名称
        custom_prompt (str): 自定义提示
        special_terms (str): 特殊术语
        
    返回:
        dict: 处理结果
    """
    try:
        # 使用带重试功能的转写函数
        transcription = robust_transcribe(audio_path, max_attempts=3)
                
        # 继续后续处理...
        # ...
        
        # 后续代码保持不变
        # ...
        
    except Exception as e:
        logger.error(f"处理音频失败: {str(e)}", exc_info=True)
        raise 


async def test_deepseek_connection(verbose=True):
    """
    测试与DeepSeek API的连接并生成全面诊断报告
    
    参数:
        verbose (bool): 是否打印详细信息
        
    返回:
        dict: 连接测试结果
    """
    logger.info("开始测试DeepSeek API连接...")
    
    results = {
        "api_key_check": None,
        "network_diagnosis": None,
        "api_test": None,
        "summary": None
    }
    
    # 1. 检查API密钥是否存在
    if not DEEPSEEK_API_KEY:
        logger.error("DeepSeek API密钥未设置")
        results["api_key_check"] = {
            "success": False,
            "error": "API密钥未设置"
        }
    elif len(DEEPSEEK_API_KEY) < 20:  # 假设有效的API密钥至少有20个字符
        logger.warning(f"DeepSeek API密钥可能无效: 长度为{len(DEEPSEEK_API_KEY)}个字符")
        results["api_key_check"] = {
            "success": False,
            "error": f"API密钥长度异常，仅有{len(DEEPSEEK_API_KEY)}个字符"
        }
    else:
        # 检查API密钥格式
        import re
        valid_format = re.match(r'^sk-[a-zA-Z0-9]{24,}$', DEEPSEEK_API_KEY)
        if not valid_format:
            logger.warning(f"DeepSeek API密钥格式可能不正确（应以'sk-'开头）")
            results["api_key_check"] = {
                "success": False,
                "error": "API密钥格式可能不正确（应以'sk-'开头）"
            }
        else:
            logger.info("DeepSeek API密钥格式正确")
            results["api_key_check"] = {
                "success": True
            }
    
    # 2. 进行网络诊断
    try:
        host = "api.deepseek.com"
        logger.info(f"开始对 {host} 进行网络诊断...")
        
        # 执行简单的网络连接测试
        import socket
        try:
            socket.getaddrinfo(host, 443)
            logger.info(f"DNS解析 {host} 成功")
            results["network_diagnosis"] = {"success": True}
        except Exception as e:
            logger.error(f"DNS解析失败: {str(e)}")
            results["network_diagnosis"] = {
                "success": False,
                "error": f"DNS解析失败: {str(e)}"
            }
            
    except Exception as e:
        logger.error(f"网络诊断失败: {str(e)}")
        results["network_diagnosis"] = {
            "success": False,
            "error": str(e)
        }
    
    # 3. 尝试发送一个简单的API请求
    try:
        logger.info("尝试发送简单的API请求...")
        
        # 创建客户端 - 使用浮点数作为timeout
        timeout_seconds = 30.0  # 设置30秒超时
        client = AsyncOpenAI(
            api_key=DEEPSEEK_API_KEY, 
            base_url="https://api.deepseek.com",
            timeout=timeout_seconds  # 使用浮点数
        )
        
        logger.info(f"创建测试客户端: timeout={timeout_seconds}秒")
        
        # 发送一个简单的请求
        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个有用的助手。"},
                {"role": "user", "content": "你好，这是一个连接测试。"}
            ],
            max_tokens=10  # 限制返回的token数量，加快测试
        )
        
        # 记录成功
        logger.info(f"API请求成功: {response.model_dump_json(indent=2)}")
        results["api_test"] = {
            "success": True,
            "response": response.model_dump()
        }
    except Exception as e:
        logger.error(f"API请求失败: {str(e)}")
        
        # 获取详细错误信息
        import traceback
        stack_trace = traceback.format_exc()
        
        results["api_test"] = {
            "success": False,
            "error": str(e),
            "traceback": stack_trace
        }
        
        # 检查是否为连接错误
        if isinstance(e, openai.APIConnectionError):
            logger.error("这是一个连接错误，可能是网络问题")
            
            # 记录底层原因
            if hasattr(e, '__cause__') and e.__cause__:
                logger.error(f"底层原因: {type(e.__cause__).__name__}: {str(e.__cause__)}")
        
        # 检查是否为授权错误
        elif isinstance(e, openai.AuthenticationError):
            logger.error("这是一个授权错误，API密钥可能无效")
    
    # 4. 生成摘要
    success_count = sum(1 for key, value in results.items() 
                       if isinstance(value, dict) and value.get("success") == True)
    total_tests = sum(1 for key, value in results.items() 
                     if isinstance(value, dict) and "success" in value)
    
    if total_tests > 0:
        success_rate = success_count / total_tests
    else:
        success_rate = 0
    
    if success_rate == 1.0:
        summary = "所有测试都通过，DeepSeek API连接正常。"
    elif success_rate > 0.5:
        summary = "部分测试通过，DeepSeek API连接可能有轻微问题。"
    else:
        summary = "大多数测试失败，DeepSeek API连接存在严重问题。"
    
    results["summary"] = {
        "success_rate": success_rate,
        "conclusion": summary
    }
    
    logger.info(f"DeepSeek API连接测试完成: {summary}")
    
    return results 