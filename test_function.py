import os
print(f"环境变量中的API密钥: {os.environ.get('OPENAI_API_KEY')[:5]}...{os.environ.get('OPENAI_API_KEY')[-4:]}")
