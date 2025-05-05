import os
import base64
import requests
from loguru import logger
from pathlib import Path
from joblib import Parallel, delayed

def to_b64(file_path):
    """将文件转换为Base64格式"""
    try:
        with open(file_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        raise Exception(f'File: {file_path} - Info: {e}')

def do_parse(file_path, 
             url='http://127.0.0.1:8000/predict',  # 修正url
             base_input_dir='/home/xinyaoh/minus_t/test1',  # 输入根目录
             **kwargs):
    """处理单个文件并发送请求到服务端"""
    if not os.path.isfile(file_path):
        raise ValueError(f"Invalid file path: {file_path}.")
    
    try:
        # 生成原始文件名
        original_filename = Path(file_path).stem
        
        # 计算相对路径，基于输入目录
        relative_path = os.path.relpath(file_path, base_input_dir)
        
        # 直接将输出文件保存到相同位置
        # output_dir = os.path.dirname(file_path)
        output_dir = "/home/xinyaoh/outputs/mineru_xyh_large_1.3.10"
        
        # 发送请求到服务端
        response = requests.post(url, json={
            'file': to_b64(file_path),
            'output_dir': output_dir,  # 输出文件保存到相同路径
            'original_filename': original_filename,
            'kwargs': kwargs
        })
        
        if response.status_code == 200:
            output = response.json()
            output['file_path'] = file_path
            return output
        else:
            raise Exception(response.text)
    except Exception as e:
        logger.error(f'File: {file_path} - Info: {e}')

def process_files(input_dirs, n_jobs=8):
    """递归遍历多个文件夹，处理所有支持的文件类型"""
    # 支持的文件扩展名
    supported_extensions = ['.pdf', '.jpg', '.png', '.doc', '.docx', '.ppt', '.pptx']
    
    # 存储所有符合条件的文件路径
    files_to_process = []
    
    # 遍历文件夹列表中的所有文件
    for input_dir in input_dirs:
        if not os.path.isdir(input_dir):
            raise ValueError(f"Invalid directory: {input_dir}")
        
        for root, dirs, files in os.walk(input_dir):
            # 跳过名为 'auto' 的文件夹
            if 'auto' in dirs:
                dirs.remove('auto')  # 从 dirs 列表中移除 'auto' 文件夹
            
            for file in files:
                if any(file.endswith(ext) for ext in supported_extensions):
                    files_to_process.append(os.path.join(root, file))
    
    # 确定并行处理的数量
    n_jobs = min(len(files_to_process), n_jobs)
    
    # 执行并行任务，处理文件
    results = Parallel(n_jobs=n_jobs, prefer='threads', verbose=10)( 
        delayed(do_parse)(p, base_input_dir=input_dir) for p in files_to_process
    )
    return results

if __name__ == '__main__':
    # 设置输入文件夹列表
    input_dirs =["/home/xinyaoh/inputs/pdf4"]
    
    # 处理文件并输出结果
    results = process_files(input_dirs, n_jobs=4)
    
    # 打印结果
    # print(results)