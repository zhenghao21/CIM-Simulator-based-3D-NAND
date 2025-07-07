from GPT2 import HardwareAcceleratedGPT2
import config
from hardware_visualizer import HardwareVisualizer
from PyQt6.QtWidgets import QApplication
import sys
import logging
from datetime import datetime
import os
import json
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，适合在多线程环境下使用
# 配置matplotlib
matplotlib.rcParams['font.family'] = 'Arial'  # 设置Arial字体
matplotlib.rcParams['font.weight'] = 'bold'   # 设置字体加粗
matplotlib.rcParams['font.size'] = 10         # 设置字体大小为10
matplotlib.rcParams['axes.unicode_minus'] = False  # 保持负号显示正常

# 添加坐标轴设置
matplotlib.rcParams['axes.spines.left'] = True          # 显示左侧坐标轴线
matplotlib.rcParams['axes.spines.bottom'] = True        # 显示底部坐标轴线
matplotlib.rcParams['axes.spines.top'] = True          # 显示顶部坐标轴线
matplotlib.rcParams['axes.spines.right'] = True        # 显示右侧坐标轴线
matplotlib.rcParams['axes.linewidth'] = 2.0
# 添加刻度线设置
matplotlib.rcParams['xtick.direction'] = 'in'  # x轴刻度线向内
matplotlib.rcParams['ytick.direction'] = 'in'  # y轴刻度线向内
matplotlib.rcParams['xtick.minor.visible'] = True  # 显示x轴次刻度线
matplotlib.rcParams['ytick.minor.visible'] = True  # 显示y轴次刻度线
matplotlib.rcParams['xtick.top'] = True               # 是否在顶部显示x轴刻度线
matplotlib.rcParams['ytick.right'] = True             # 是否在右侧显示y轴刻度线
# 设置图形保存分辨率
matplotlib.rcParams['savefig.dpi'] = 600
# 设置图形大小和分辨率
matplotlib.rcParams['figure.figsize'] = [5,5]     # 图形大小，单位英寸
matplotlib.rcParams['figure.dpi'] = 600                # 图形分辨率
def run_with_param_value(param_name, param_value, current_time):
    """
    使用指定的参数名和参数值运行模型
    
    参数:
    - param_name: 要修改的config参数名称
    - param_value: 要设置的参数值
    - prompt: 生成文本的提示
    
    返回:
    - 生成的文本
    - 总计算时间
    """
    # 动态修改config模块中的参数
    if isinstance(param_name,list):
        for index,name in enumerate(param_name):
            setattr(config, name, param_value[index])
    else:
        setattr(config, param_name, param_value)

    
    log_dir = config.CONFIG_PATH + "/" + current_time
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(
        log_dir, 
        f"计算日志.log"
    )
    
    # 配置日志记录器
    logger = logging.getLogger(f"{current_time}")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []  # 清除之前的处理器
    
    # 添加文件处理器（记录所有信息）
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # 添加控制台处理器（只显示重要信息）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    
    # 将处理器添加到记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"配置路径: {config.CONFIG_PATH}")
    
    # 保存当前配置到日志目录
    with open(os.path.join(log_dir, 'config_snapshot.json'), 'w') as f:
        config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('__')}
        # 处理不可序列化的值
        for k, v in config_dict.items():
            if not isinstance(v, (str, int, float, bool, list, dict, type(None))):
                config_dict[k] = str(v)
        json.dump(config_dict, f, indent=2)
    
    # 创建GPT2模型实例
    logger.info("初始化硬件加速的GPT-2模型...")
    model = HardwareAcceleratedGPT2(logger, log_dir)
    
    if config.VISUALIZE:
        app = QApplication(sys.argv)
        ex = HardwareVisualizer(model.hardware_sim.cell_array)
        ex.show()
    
    # 生成文本
    logger.info(f"使用提示: '{config.PROMPT}'")
    generated_text, total_time = model.generate(
        prompt=config.PROMPT,
        max_length=config.DEFAULT_MAX_LENGTH,
        temperature=config.DEFAULT_TEMPERATURE,
        top_k=config.DEFAULT_TOP_K
    )
    
    # 保存结果到文件
    with open(os.path.join(log_dir, 'results.json'), 'w') as f:
        results = {
            'prompt': config.PROMPT,
            'generated_text': generated_text,
            'total_time': total_time,
        }
        json.dump(results, f, indent=2)

def main(config_list):
    # 定义要测试的参数及其值列表
    # param_variations = {
    #     # 'QUANTIZATION_BITS': [4,8,16],
    #     # 'BL_PER_OPERATION': [128],
    #     # 'MAX_CURRENT': [ 160],
    #     # 'CURRENT_VARIATION_MODE': [True],
    #     # 'SCALE':[0],
    #     'BLOCKS_PER_OPERATION':[48],
    #     'DY_WEIGHT':[True,False]#是否启用动态权重
    #     # 'SYMMETRIC':[True]#是否启用对称性
    # }
    # keys = list(param_variations.keys())
    # values = list(param_variations.values())
    
    # # 生成所有可能的组合
    # combinations = list(itertools.product(*values))
    # # 使用的提示文本
    # prompt = "how are you going"
    
    
    # 为每个参数值运行模型
    for config in config_list:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_with_param_value(list(config.keys()) , list(config.values()),current_time)
