import re
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
import traceback
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import torch

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

class GPT2PowerAnalyzer:
    """
    GPT-2模型功耗分析器
    用于解析日志文件并分析硬件加速功耗消耗
    """
    
    def __init__(self, log_file, start_step=None, end_step=None):
        """
        初始化分析器
        
        参数:
        - log_file: 日志文件路径
        - start_step: 开始分析的步骤 (可选)
        - end_step: 结束分析的步骤 (可选)
        """
        self.log_file = log_file
        self.start_step = start_step
        self.end_step = end_step
        
        self.power_data = { 
            'qkv_power': defaultdict(list),       # QKV计算功耗
            'proj_power': defaultdict(list),      # 投影计算功耗
            'fc1_power': defaultdict(list),       # MLP第一层功耗
            'fc2_power': defaultdict(list),       # MLP第二层功耗
            'dequant_power': defaultdict(list), # 反量化的功耗
            'layer_power': defaultdict(list),     # 每层总功耗
            'qkv_sensing': defaultdict(list),     # QKV的sensing功耗
            'proj_sensing': defaultdict(list),    # 投影的sensing功耗
            'fc1_sensing': defaultdict(list),     # MLP第一层的sensing功耗
            'fc2_sensing': defaultdict(list),     # MLP第二层的sensing功耗
            'dequant_sensing': defaultdict(list),   # 反量化的sensing功耗
        }
        self.total_power = 0                      # 总功耗
        self.total_sensing_power = 0              # 总sensing功耗
        self.layer_count = 0                      # 层数
        self.power_per_layer = {}                 # 每层功耗统计
        
        # 操作类型映射表（中文到英文）
        self.operation_map = {
            'QKV计算': 'QKV Computation',
            '注意力输出投影': 'Attention Projection',
            'MLP第一层': 'MLP First Layer',
            'MLP第二层': 'MLP Second Layer',
            '反量化': 'Dequantization',
            '总功耗': 'Total Power',
            'Sensing功耗': 'Sensing Power'
        }
        
        # 自定义颜色方案 - 使用更鲜明的颜色
        self.pie_colors = [
            '#FF5252', '#FF9D52', '#FFD152', 
            '#66BB6A', '#42A5F5', '#7E57C2', 
            '#EC407A', '#26C6DA', '#5C6BC0'
        ]
        
        # 加载并解析日志文件
        self._parse_log_file()

    def _parse_log_file(self):
        """解析日志文件，提取功耗信息，支持按步骤范围过滤"""
        print(f"解析日志文件: {self.log_file}")
        
        # 首先读取完整日志文件
        try:
            with open(self.log_file, 'r', encoding='gbk', errors='ignore') as f:
                log_content = f.read()
        except UnicodeDecodeError:
            # 如果gbk解码失败，尝试utf-8编码
            with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
                log_content = f.read()
                
        # 如果指定了步骤范围，则提取对应范围内的日志部分
        if self.start_step is not None or self.end_step is not None:
            # 查找所有步骤的开始位置
            step_pattern = r"开始生成步骤(\d+)的文本"
            step_matches = re.finditer(step_pattern, log_content)
            
            # 收集步骤信息
            step_positions = []
            for match in step_matches:
                step_num = int(match.group(1))
                step_positions.append((step_num, match.start()))
            
            if not step_positions:
                print("警告: 日志中未找到步骤信息，将分析整个日志文件")
            else:
                # 对步骤按索引排序
                step_positions.sort()
                
                # 确定分析范围的开始和结束位置
                start_pos = 0
                end_pos = len(log_content)
                
                # 过滤步骤
                filtered_steps = []
                for step_num, pos in step_positions:
                    if (self.start_step is None or step_num >= self.start_step) and \
                       (self.end_step is None or step_num <= self.end_step):
                        filtered_steps.append((step_num, pos))
                
                if filtered_steps:
                    # 找到范围内第一个步骤的开始位置
                    start_pos = filtered_steps[0][1]
                    
                    # 找到范围内最后一个步骤的结束位置（下一个步骤的开始，或文件结束）
                    last_step_num = filtered_steps[-1][0]
                    last_step_pos = filtered_steps[-1][1]
                    
                    # 查找下一个步骤的位置（如果有）
                    for step_num, pos in step_positions:
                        if step_num > last_step_num:
                            end_pos = pos
                            break
                            
                    # 仅保留指定范围内的日志内容
                    log_content = log_content[start_pos:end_pos]
                    print(f"过滤日志到步骤范围: {filtered_steps[0][0]} 到 {filtered_steps[-1][0]}")
                else:
                    print(f"警告: 在指定范围（{self.start_step} 到 {self.end_step}）内没有找到步骤，将分析整个日志文件")
        
        # 定义正则表达式模式
        power_pattern = re.compile(r'计算(.*?)，耗能([\d\.]+)uJ，目前总耗能([\d\.]+)uJ')
        sensing_pattern = re.compile(r'计算(.*?)，.*?中有([\d\.]+)uJ用于sensing，目前总sensing耗能([\d\.]+)uJ')
        layer_start_pattern = re.compile(r'开始处理第(\d+)层')
        
        current_layer = -1
        
        # 逐行解析日志内容
        for line in log_content.splitlines():
            # 检查层开始
            layer_match = layer_start_pattern.search(line)
            if layer_match:
                current_layer = int(layer_match.group(1))
                self.layer_count = max(self.layer_count, current_layer + 1)
                continue
            
            # 检查功耗计算
            power_match = power_pattern.search(line)
            if power_match:
                operation = power_match.group(1)
                power_cost = float(power_match.group(2))
                total_power = float(power_match.group(3))
                self.total_power = max(self.total_power, total_power)
                if "反量化" in operation:
                    self.power_data['dequant_power'][current_layer].append(power_cost)
                elif "QKV" in operation:
                    self.power_data['qkv_power'][current_layer].append(power_cost)
                elif "注意力输出投影" in operation:
                    self.power_data['proj_power'][current_layer].append(power_cost)
                elif "MLP第一层" in operation:
                    self.power_data['fc1_power'][current_layer].append(power_cost)
                elif "MLP第二层" in operation:
                    self.power_data['fc2_power'][current_layer].append(power_cost)

                continue
            
            # 检查sensing功耗
            sensing_match = sensing_pattern.search(line)
            if sensing_match:
                operation = sensing_match.group(1)
                sensing_cost = float(sensing_match.group(2))
                total_sensing = float(sensing_match.group(3))
                self.total_sensing_power = max(self.total_sensing_power, total_sensing)
                if "反量化" in operation:
                    self.power_data['dequant_sensing'][current_layer].append(sensing_cost)
                elif "QKV" in operation:
                    self.power_data['qkv_sensing'][current_layer].append(sensing_cost)
                elif "注意力输出投影" in operation:
                    self.power_data['proj_sensing'][current_layer].append(sensing_cost)
                elif "MLP第一层" in operation:
                    self.power_data['fc1_sensing'][current_layer].append(sensing_cost)
                elif "MLP第二层" in operation:
                    self.power_data['fc2_sensing'][current_layer].append(sensing_cost)

                continue
        
        # 计算每层总功耗
        for layer in range(self.layer_count):
            layer_total = 0
            for op_type in ['qkv_power', 'proj_power', 'fc1_power', 'fc2_power','dequant_power']:
                if layer in self.power_data[op_type]:
                    layer_total += sum(self.power_data[op_type][layer])
            self.power_data['layer_power'][layer] = [layer_total]
            self.power_per_layer[layer] = layer_total
            
        print(f"解析完成，发现 {self.layer_count} 层")
        
        # 添加步骤范围信息
        if self.start_step is not None or self.end_step is not None:
            range_info = f"步骤范围: {self.start_step if self.start_step is not None else '开始'} 到 {self.end_step if self.end_step is not None else '结束'}"
            print(range_info)
            
        # 打印功耗统计摘要
        print(f"总功耗: {self.total_power:.2f} uJ")
        print(f"总Sensing功耗: {self.total_sensing_power:.2f} uJ")

    def analyze_by_operation(self):
        """按操作类型分析功耗消耗"""
        operation_power = {
            'QKV计算': sum(sum(powers) for powers in self.power_data['qkv_power'].values()),
            '注意力输出投影': sum(sum(powers) for powers in self.power_data['proj_power'].values()),
            'MLP第一层': sum(sum(powers) for powers in self.power_data['fc1_power'].values()),
            'MLP第二层': sum(sum(powers) for powers in self.power_data['fc2_power'].values()),
            '反量化': sum(sum(powers) for powers in self.power_data['dequant_power'].values())
        }
        
        return operation_power
    
    def analyze_sensing_by_operation(self):
        """按操作类型分析sensing功耗消耗"""
        sensing_power = {
            'QKV计算': sum(sum(powers) for powers in self.power_data['qkv_sensing'].values()),
            '注意力输出投影': sum(sum(powers) for powers in self.power_data['proj_sensing'].values()),
            'MLP第一层': sum(sum(powers) for powers in self.power_data['fc1_sensing'].values()),
            'MLP第二层': sum(sum(powers) for powers in self.power_data['fc2_sensing'].values()),
            '反量化': sum(sum(powers) for powers in self.power_data['dequant_sensing'].values())
        }
        
        return sensing_power
    
    def analyze_by_layer(self):
        """按层分析功耗消耗"""
        layer_power = {}
        for layer in range(self.layer_count):
            if layer in self.power_data['layer_power']:
                layer_power[f"层 {layer}"] = self.power_data['layer_power'][layer][0]
        
        return layer_power
    
    def analyze_layer_composition(self, layer_num):
        """分析指定层的功耗组成"""
        if layer_num not in range(self.layer_count):
            print(f"错误: 层 {layer_num} 不存在")
            return {}
            
        layer_composition = {
            'QKV计算': sum(self.power_data['qkv_power'].get(layer_num, [])),
            '注意力输出投影': sum(self.power_data['proj_power'].get(layer_num, [])),
            'MLP第一层': sum(self.power_data['fc1_power'].get(layer_num, [])),
            'MLP第二层': sum(self.power_data['fc2_power'].get(layer_num, [])),
            '反量化': sum(self.power_data['dequant_power'].get(layer_num, []))

        }
        
        return layer_composition
    
    def analyze_sensing_layer_composition(self, layer_num):
        """分析指定层的sensing功耗组成"""
        if layer_num not in range(self.layer_count):
            print(f"错误: 层 {layer_num} 不存在")
            return {}
            
        sensing_composition = {
            'QKV计算': sum(self.power_data['qkv_sensing'].get(layer_num, [])),
            '注意力输出投影': sum(self.power_data['proj_sensing'].get(layer_num, [])),
            'MLP第一层': sum(self.power_data['fc1_sensing'].get(layer_num, [])),
            'MLP第二层': sum(self.power_data['fc2_sensing'].get(layer_num, [])),
            '反量化': sum(self.power_data['dequant_sensing'].get(layer_num, []))
        }
        
        return sensing_composition
        
    def analyze_by_step(self, start_step=None, end_step=None):
        """按步骤分析功耗消耗，可以指定起始和结束步骤"""
        # 如果未指定步骤范围，则使用初始化时的步骤范围
        start_step = start_step if start_step is not None else self.start_step
        end_step = end_step if end_step is not None else self.end_step
        
        step_power = {}
        step_sensing_power = {}
        
        # 读取日志文件内容
        try:
            with open(self.log_file, 'r', encoding='gbk') as f:
                log_text = f.read()
        except UnicodeDecodeError:
            # 如果gbk解码失败，尝试其他编码
            with open(self.log_file, 'r', encoding='utf-8') as f:
                log_text = f.read()
        
        # 首先尝试从日志中提取步骤信息
        step_pattern = r"开始生成步骤(\d+)的文本"
        step_matches = re.findall(step_pattern, log_text)
        
        if not step_matches:
            print("未在日志中找到生成步骤信息")
            return {}, {}
        
        # 获取所有出现的步骤，并转换为整数
        all_steps = sorted([int(step) for step in step_matches])
        
        # 如果指定了开始和结束步骤，则过滤步骤
        filtered_steps = all_steps.copy()
        if start_step is not None:
            filtered_steps = [s for s in filtered_steps if s >= start_step]
        if end_step is not None:
            filtered_steps = [s for s in filtered_steps if s <= end_step]
            
        if not filtered_steps:
            print(f"在指定范围内（{start_step if start_step is not None else '开始'} 到 {end_step if end_step is not None else '结束'}）没有找到步骤")
            return {}, {}
        
        print(f"分析步骤范围: {filtered_steps[0]} 到 {filtered_steps[-1]}")
        
        # 对每个步骤进行功耗分析
        for i, step in enumerate(filtered_steps):
            # 查找该步骤的开始位置
            step_start_pattern = f"开始生成步骤{step}的文本"
            step_start = log_text.find(step_start_pattern)
            
            if step_start == -1:
                print(f"无法找到步骤 {step} 的开始位置")
                continue
                
            # 移动到步骤标记之后
            step_start = step_start + len(step_start_pattern)
            
            # 找下一步的开始位置，或者文件结束
            next_step_start = len(log_text)
            
            if i < len(filtered_steps) - 1:
                next_step = filtered_steps[i + 1]
                next_step_pattern = f"开始生成步骤{next_step}的文本"
                next_pos = log_text.find(next_step_pattern, step_start)
                if next_pos != -1:
                    next_step_start = next_pos
            
            # 提取该步骤的日志内容
            step_log = log_text[step_start:next_step_start]
            
            # 计算该步骤的总功耗
            power_pattern = re.compile(r'计算(.*?)，耗能([\d\.]+)uJ，目前总耗能([\d\.]+)uJ')
            power_matches = power_pattern.findall(step_log)
            step_power[step] = sum(float(match[1]) for match in power_matches) if power_matches else 0
            
            # 计算该步骤的sensing功耗
            sensing_pattern = re.compile(r'计算(.*?)，.*?中有([\d\.]+)uJ用于sensing，目前总sensing耗能([\d\.]+)uJ')
            sensing_matches = sensing_pattern.findall(step_log)
            step_sensing_power[step] = sum(float(match[1]) for match in sensing_matches) if sensing_matches else 0
        
        if not step_power:
            print(f"在指定范围内（{start_step if start_step is not None else '开始'} 到 {end_step if end_step is not None else '结束'}）没有找到功耗数据")
            
        return step_power, step_sensing_power

    def plot_step_power(self, save_path='step_power_distribution.png', save_data=True, start_step=None, end_step=None):
        """使用柱状图展示每个生成步骤的功耗分布"""
        step_power, step_sensing_power = self.analyze_by_step(start_step, end_step)
        
        if not step_power:
            print("没有步骤功耗数据可供分析")
            return None
        
        # 构建范围描述，用于图表标题
        range_desc = ""
        if start_step is not None and end_step is not None:
            range_desc = f" (步骤 {start_step}~{end_step})"
        elif start_step is not None:
            range_desc = f" (步骤 {start_step}及之后)"
        elif end_step is not None:
            range_desc = f" (步骤 0~{end_step})"
        
        plt.figure(figsize=(12, 8))
        
        # 准备数据
        steps = sorted(step_power.keys())
        power_values = [step_power[step] for step in steps]
        sensing_values = [step_sensing_power[step] for step in steps]
        
        # 创建柱状图
        bar_width = 0.35
        index = np.arange(len(steps))
        
        plt.bar(index, power_values, bar_width, label='总功耗', color='blue')
        plt.bar(index + bar_width, sensing_values, bar_width, label='Sensing功耗', color='green')
        
        plt.xlabel('生成步骤')
        plt.ylabel('功耗 (uJ)')
        plt.title(f'GPT-2模型每个生成步骤的功耗分布{range_desc}')
        plt.xticks(index + bar_width/2, [f'步骤 {step}' for step in steps])
        plt.legend()
        
        # 添加数值标签
        for i, v in enumerate(power_values):
            plt.text(i - 0.05, v + 0.1, f'{v:.2f}', rotation=45)
        
        for i, v in enumerate(sensing_values):
            plt.text(i + bar_width - 0.05, v + 0.1, f'{v:.2f}', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"步骤功耗分布柱状图已保存为 '{save_path}'")
        
        # 保存数据
        if save_data:
            # 创建数据字典
            data_dict = {}
            for step in steps:
                data_dict[f'步骤 {step} 总功耗'] = step_power[step]
                data_dict[f'步骤 {step} Sensing功耗'] = step_sensing_power[step]
            
            data_path = os.path.splitext(save_path)[0] + "_data.csv"
            self.save_data_to_csv(data_dict, data_path)
            
            excel_path = os.path.splitext(save_path)[0] + "_data.xlsx"
            self.save_data_to_excel(data_dict, excel_path)
        
        return save_path

    def save_data_to_csv(self, data_dict, save_path):
        """将数据保存为CSV文件"""
        df = pd.DataFrame(list(data_dict.items()), columns=['Name', 'Value'])
        df.to_csv(save_path, index=False, encoding='gbk')
        print(f"数据已保存到: {save_path}")
        return save_path

    def save_data_to_excel(self, data_dict, save_path):
        """将数据保存为Excel文件"""
        df = pd.DataFrame(list(data_dict.items()), columns=['Name', 'Value'])
        df.to_excel(save_path, index=False, engine='openpyxl')
        print(f"数据已保存到: {save_path}")
        return save_path

    def plot_power_by_operation(self, save_path='operation_power_distribution.png', save_data=True):
        """使用增强型饼图创建3D效果的操作功耗分布图"""
        operation_power = self.analyze_by_operation()
        
        # 创建英文键字典
        eng_operation_power = {}
        for k, v in operation_power.items():
            eng_operation_power[self.operation_map.get(k, k)] = v
        
        # 计算每个扇区的偏移量，使数值大的扇区凸出
        values = list(eng_operation_power.values())
        labels = list(eng_operation_power.keys())
        total = sum(values)
        explode = [0.05 + 0.05 * (v / max(values)) for v in values]
        
        # 准备包含百分比和功耗的标签
        labels_with_values = []
        for i, label in enumerate(labels):
            pct = values[i] / total * 100
            labels_with_values.append(f'{label}\n{pct:.1f}% ({values[i]:.2f} uJ)')
        
        # 创建饼图 (使用增强效果模拟3D)
        plt.figure(figsize=(12, 9))
        wedges, texts = plt.pie(
            values, 
            labels=labels,  # 不在饼图上显示标签
            explode=explode,  # 突出显示各个扇区
            autopct=None,  # 不在饼图上显示百分比
            startangle=140,   # 旋转饼图以获得更好的视觉效果
            colors=self.pie_colors[:len(values)],    
            wedgeprops={'edgecolor': 'black', 'linewidth': 1.2, 'antialiased': True}  # 添加黑色边框
        )
        
        # 构建标题，包含步骤范围信息
        title = "Operation Types"
        if self.start_step is not None or self.end_step is not None:
            range_str = f" (Steps {self.start_step if self.start_step is not None else 'Start'} to {self.end_step if self.end_step is not None else 'End'})"
            title += range_str
        
        # 添加图例，包含标签、百分比和功耗
        plt.legend(
            wedges, 
            labels_with_values,
            title=title,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            fontsize=10
        )
        plt.axis('equal')  # 确保饼图是圆形的
        
        # 保存图表
        plt.tight_layout()
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"操作功耗分布3D风格饼图已保存为 '{save_path}'")
        
        # 保存数据
        if save_data:
            data_path = os.path.splitext(save_path)[0] + "_data.csv"
            self.save_data_to_csv(eng_operation_power, data_path)
            
            excel_path = os.path.splitext(save_path)[0] + "_data.xlsx"
            self.save_data_to_excel(eng_operation_power, excel_path)
            
        return save_path
    
    def plot_sensing_power_by_operation(self, save_path='sensing_power_distribution.png', save_data=True):
        """使用增强型饼图创建3D效果的sensing操作功耗分布图"""
        sensing_power = self.analyze_sensing_by_operation()
        
        # 创建英文键字典
        eng_sensing_power = {}
        for k, v in sensing_power.items():
            eng_sensing_power[self.operation_map.get(k, k)] = v
        
        # 计算每个扇区的偏移量，使数值大的扇区凸出
        values = list(eng_sensing_power.values())
        labels = list(eng_sensing_power.keys())
        total = sum(values)
        explode = [0.05 + 0.05 * (v / max(values)) for v in values]
        
        # 准备包含百分比和功耗的标签
        labels_with_values = []
        for i, label in enumerate(labels):
            pct = values[i] / total * 100
            labels_with_values.append(f'{label}\n{pct:.1f}% ({values[i]:.2f} uJ)')
        
        # 创建饼图 (使用增强效果模拟3D)
        plt.figure(figsize=(12, 9))
        wedges, texts = plt.pie(
            values, 
            labels=labels,  # 不在饼图上显示标签
            explode=explode,  # 突出显示各个扇区
            autopct=None,  # 不在饼图上显示百分比
            startangle=140,   # 旋转饼图以获得更好的视觉效果
            colors=self.pie_colors[:len(values)],    
            wedgeprops={'edgecolor': 'black', 'linewidth': 1.2, 'antialiased': True}  # 添加黑色边框
        )
        
        # 构建标题，包含步骤范围信息
        title = "Operation Types (Sensing)"
        if self.start_step is not None or self.end_step is not None:
            range_str = f" (Steps {self.start_step if self.start_step is not None else 'Start'} to {self.end_step if self.end_step is not None else 'End'})"
            title += range_str
        
        # 添加图例，包含标签、百分比和功耗
        plt.legend(
            wedges, 
            labels_with_values,
            title=title,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            fontsize=10
        )
        plt.axis('equal')  # 确保饼图是圆形的
        
        # 保存图表
        plt.tight_layout()
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"Sensing功耗分布3D风格饼图已保存为 '{save_path}'")
        
        # 保存数据
        if save_data:
            data_path = os.path.splitext(save_path)[0] + "_data.csv"
            self.save_data_to_csv(eng_sensing_power, data_path)
            
            excel_path = os.path.splitext(save_path)[0] + "_data.xlsx"
            self.save_data_to_excel(eng_sensing_power, excel_path)
            
        return save_path
    
    def plot_power_by_layer(self, save_path='layer_power_distribution.png', save_data=True):
        """使用柱状图展示每层功耗分布"""
        layer_power = self.analyze_by_layer()
        
        # 翻译层名称
        eng_layer_power = {}
        for k, v in layer_power.items():
            layer_num = k.split(' ')[1]  # 提取数字部分
            eng_layer_power[f"Layer {layer_num}"] = v
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(eng_layer_power)), eng_layer_power.values(), color='green')
        plt.xlabel('Layer Index')
        plt.ylabel('Power (uJ)')
        
        # 添加步骤范围到标题
        title = 'Power Consumption per Layer in GPT-2 Model'
        if self.start_step is not None or self.end_step is not None:
            range_str = f" (Steps {self.start_step if self.start_step is not None else 'Start'} to {self.end_step if self.end_step is not None else 'End'})"
            title += range_str
        plt.title(title)
        
        plt.xticks(range(len(eng_layer_power)), eng_layer_power.keys())
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    '%.2f' % height,
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"层功耗分布柱状图已保存为 '{save_path}'")
        
        # 保存数据
        if save_data:
            data_path = os.path.splitext(save_path)[0] + "_data.csv"
            self.save_data_to_csv(eng_layer_power, data_path)
            
            excel_path = os.path.splitext(save_path)[0] + "_data.xlsx"
            self.save_data_to_excel(eng_layer_power, excel_path)
            
        return save_path
    
    def plot_layer_composition(self, layer_num, save_path=None, save_data=True):
        """使用增强型饼图创建3D效果的层功耗组成图"""
        if save_path is None:
            save_path = f'layer_{layer_num}_power_composition.png'
            
        layer_comp = self.analyze_layer_composition(layer_num)
        
        # 创建英文键字典
        eng_layer_comp = {}
        for k, v in layer_comp.items():
            eng_layer_comp[self.operation_map.get(k, k)] = v
        
        # 过滤零值
        eng_layer_comp = {k: v for k, v in eng_layer_comp.items() if v > 0}
        
        # 计算每个扇区的偏移量，使数值大的扇区凸出
        values = list(eng_layer_comp.values())
        labels = list(eng_layer_comp.keys())
        
        # 确保值不全为0
        if not values or sum(values) == 0:
            print(f"第 {layer_num} 层没有数据可显示")
            return
            
        total = sum(values)
        explode = [0.05 + 0.05 * (v / max(values)) for v in values]
        
        # 准备包含百分比和功耗的标签
        labels_with_values = []
        for i, label in enumerate(labels):
            pct = values[i] / total * 100
            labels_with_values.append(f'{label}\n{pct:.1f}% ({values[i]:.2f} uJ)')
        
        # 创建饼图 (使用增强效果模拟3D)
        plt.figure(figsize=(12, 9))
        wedges, texts = plt.pie(
            values, 
            labels=labels,  # 不在饼图上显示标签
            explode=explode,  # 突出显示各个扇区
            autopct=None,  # 不在饼图上显示百分比
            startangle=140,   # 旋转饼图以获得更好的视觉效果
            colors=self.pie_colors[:len(values)],
            wedgeprops={'edgecolor': 'black', 'linewidth': 1.2, 'antialiased': True}  # 添加黑色边框
        )
        
        # 构建标题，包含步骤范围信息
        title = f"Layer {layer_num} Operations"
        if self.start_step is not None or self.end_step is not None:
            range_str = f" (Steps {self.start_step if self.start_step is not None else 'Start'} to {self.end_step if self.end_step is not None else 'End'})"
            title += range_str
        
        # 添加图例，包含标签、百分比和功耗
        plt.legend(
            wedges, 
            labels_with_values,
            title=title,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            fontsize=10
        )
            
        plt.axis('equal')  # 确保饼图是圆形的
        
        # 保存图表
        plt.tight_layout()
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"第 {layer_num} 层功耗组成3D风格饼图已保存为 '{save_path}'")
        
        # 保存数据
        if save_data:
            data_path = os.path.splitext(save_path)[0] + "_data.csv"
            self.save_data_to_csv(eng_layer_comp, data_path)
            
            excel_path = os.path.splitext(save_path)[0] + "_data.xlsx"
            self.save_data_to_excel(eng_layer_comp, excel_path)
            
        return save_path
        
    def plot_sensing_vs_total(self, save_path='sensing_vs_total_power.png', save_data=True):
        """创建感知功耗与总功耗比较的饼图"""
        sensing_total = self.total_sensing_power
        non_sensing_total = self.total_power - sensing_total
        
        values = [sensing_total, non_sensing_total]
        labels = ['Sensing Power', 'Other Power']
        
        # 创建数据字典
        data_dict = {'Sensing Power': sensing_total, 'Other Power': non_sensing_total}
        
        # 创建饼图
        plt.figure(figsize=(10, 8))
        plt.pie(
            values, 
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=['#66BB6A', '#42A5F5'],
            wedgeprops={'edgecolor': 'black', 'linewidth': 1.2, 'antialiased': True}
        )
        
        plt.axis('equal')
        
        # 添加步骤范围到标题
        title = 'Sensing Power vs Total Power Distribution'
        if self.start_step is not None or self.end_step is not None:
            range_str = f" (Steps {self.start_step if self.start_step is not None else 'Start'} to {self.end_step if self.end_step is not None else 'End'})"
            title += range_str
        plt.title(title)
        
        # 保存图表
        plt.tight_layout()
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"Sensing与总功耗比较饼图已保存为 '{save_path}'")
        
        # 保存数据
        if save_data:
            data_path = os.path.splitext(save_path)[0] + "_data.csv"
            self.save_data_to_csv(data_dict, data_path)
            
            excel_path = os.path.splitext(save_path)[0] + "_data.xlsx"
            self.save_data_to_excel(data_dict, excel_path)
            
        return save_path
        
    def generate_combined_report(self, output_dir):
        """生成包含所有数据的组合报告Excel文件"""
        # 构建包含步骤范围信息的文件名
        step_range_str = ""
        if self.start_step is not None and self.end_step is not None:
            step_range_str = f"_step{self.start_step}-{self.end_step}"
        elif self.start_step is not None:
            step_range_str = f"_step{self.start_step}plus"
        elif self.end_step is not None:
            step_range_str = f"_step0-{self.end_step}"
            
        report_path = os.path.join(output_dir, f"combined_power_report{step_range_str}.xlsx")
        
        # 创建一个Excel Writer对象
        with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
            # 操作功耗分布
            operation_power = self.analyze_by_operation()
            eng_operation_power = {self.operation_map.get(k, k): v for k, v in operation_power.items()}
            op_df = pd.DataFrame(list(eng_operation_power.items()), columns=['Operation', 'Power (uJ)'])
            op_df['Percentage'] = op_df['Power (uJ)'] / op_df['Power (uJ)'].sum() * 100
            op_df.to_excel(writer, sheet_name='Operation Power', index=False)
            
            # Sensing功耗分布
            sensing_power = self.analyze_sensing_by_operation()
            eng_sensing_power = {self.operation_map.get(k, k): v for k, v in sensing_power.items()}
            sensing_df = pd.DataFrame(list(eng_sensing_power.items()), columns=['Operation', 'Sensing Power (uJ)'])
            sensing_df['Percentage'] = sensing_df['Sensing Power (uJ)'] / sensing_df['Sensing Power (uJ)'].sum() * 100
            sensing_df.to_excel(writer, sheet_name='Sensing Power', index=False)
            
            # 层功耗分布
            layer_power = self.analyze_by_layer()
            eng_layer_power = {}
            for k, v in layer_power.items():
                layer_num = k.split(' ')[1]
                eng_layer_power[f"Layer {layer_num}"] = v
            layer_df = pd.DataFrame(list(eng_layer_power.items()), columns=['Layer', 'Power (uJ)'])
            layer_df.to_excel(writer, sheet_name='Layer Power', index=False)
            
            # Sensing与总功耗比较
            sensing_total = self.total_sensing_power
            non_sensing_total = self.total_power - sensing_total
            sensing_vs_total = {'Sensing Power': sensing_total, 'Other Power': non_sensing_total}
            sensing_vs_df = pd.DataFrame(list(sensing_vs_total.items()), columns=['Type', 'Power (uJ)'])
            sensing_vs_df['Percentage'] = sensing_vs_df['Power (uJ)'] / sensing_vs_df['Power (uJ)'].sum() * 100
            sensing_vs_df.to_excel(writer, sheet_name='Sensing vs Total', index=False)
            
            # 步骤分析数据
            step_power, step_sensing_power = self.analyze_by_step()
            if step_power:
                step_data = []
                for step in sorted(step_power.keys()):
                    step_data.append({
                        'Step': step,
                        'Total Power (uJ)': step_power[step],
                        'Sensing Power (uJ)': step_sensing_power[step],
                        'Sensing Percentage (%)': (step_sensing_power[step] / step_power[step] * 100) if step_power[step] > 0 else 0
                    })
                step_df = pd.DataFrame(step_data)
                step_df.to_excel(writer, sheet_name='Step Analysis', index=False)
            
            # 总结页
            # 添加步骤范围信息
            range_info = ""
            if self.start_step is not None and self.end_step is not None:
                range_info = f"步骤 {self.start_step} 到 {self.end_step}"
            elif self.start_step is not None:
                range_info = f"步骤 {self.start_step} 及之后"
            elif self.end_step is not None:
                range_info = f"步骤 0 到 {self.end_step}"
            else:
                range_info = "全部步骤"
                
            summary_data = {
                '指标': ['分析范围', '总功耗', '总Sensing功耗', 'Sensing占比', '层数', '每层平均功耗', 
                       '功耗最高层', '功耗最低层'],
                '值': [
                    range_info,
                    f"{self.total_power:.2f} uJ",
                    f"{self.total_sensing_power:.2f} uJ",
                    f"{(self.total_sensing_power/self.total_power*100):.2f}%",
                    str(self.layer_count),
                    f"{sum(operation_power.values())/self.layer_count:.2f} uJ",
                    f"Layer {max(self.power_per_layer.items(), key=lambda x: x[1])[0] if self.power_per_layer else 'N/A'}",
                    f"Layer {min(self.power_per_layer.items(), key=lambda x: x[1])[0] if self.power_per_layer else 'N/A'}"
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        print(f"组合报告已保存为: {report_path}")
        return report_path
        
    def print_power_summary(self):
        """打印功耗消耗摘要"""
        operation_power = self.analyze_by_operation()
        sensing_power = self.analyze_sensing_by_operation()
        total_op_power = sum(operation_power.values())
        total_sensing = sum(sensing_power.values())
        
        # 添加步骤范围信息
        range_info = ""
        if self.start_step is not None and self.end_step is not None:
            range_info = f"（步骤 {self.start_step} 到 {self.end_step}）"
        elif self.start_step is not None:
            range_info = f"（步骤 {self.start_step} 及之后）"
        elif self.end_step is not None:
            range_info = f"（步骤 0 到 {self.end_step}）"
        
        print(f"\n===== GPT-2 模型功耗性能分析 {range_info} =====")
        print(f"总功耗: {self.total_power:.2f} uJ")
        print(f"总Sensing功耗: {self.total_sensing_power:.2f} uJ")
        print(f"Sensing占比: {(self.total_sensing_power/self.total_power*100):.2f}%")
        
        print("\n----- 操作类型功耗分布 -----")
        for op, power in sorted(operation_power.items(), key=lambda x: x[1], reverse=True):
            eng_op = self.operation_map.get(op, op)
            print(f"{eng_op}: {power:.2f} uJ ({power/total_op_power*100:.1f}%)")
            
        print("\n----- Sensing功耗分布 -----")
        for op, power in sorted(sensing_power.items(), key=lambda x: x[1], reverse=True):
            eng_op = self.operation_map.get(op, op)
            print(f"{eng_op}: {power:.2f} uJ ({power/total_sensing*100:.1f}%)")
            
        print("\n----- 层功耗分布 TOP5 -----")
        layer_power = self.analyze_by_layer()
        for layer, power in sorted(layer_power.items(), key=lambda x: x[1], reverse=True)[:5]:
            layer_num = layer.split(' ')[1]
            print(f"Layer {layer_num}: {power:.2f} uJ")
            
        print("\n----- 硬件效率分析 -----")
        print(f"每层平均功耗: {total_op_power/self.layer_count:.2f} uJ")
        if self.power_per_layer:
            print(f"功耗最高层: Layer {max(self.power_per_layer.items(), key=lambda x: x[1])[0]}")
            print(f"功耗最低层: Layer {min(self.power_per_layer.items(), key=lambda x: x[1])[0]}")
        else:
            print("功耗最高层: N/A")
            print("功耗最低层: N/A")
        print("============================")
    
    def analyze_all(self, output_dir="power_results", detail_layers=None, save_data=True, start_step=None, end_step=None):
        """执行完整分析并生成所有图表
        
        参数:
        - output_dir: 输出图像的目录
        - detail_layers: 要详细分析的特定层列表
        - save_data: 是否保存图表数据
        - start_step: 起始分析步骤 (会覆盖初始化时的设置)
        - end_step: 结束分析步骤 (会覆盖初始化时的设置)
        """
        # 使用新的步骤范围重新解析日志文件
        if start_step is not None or end_step is not None:
            self.start_step = start_step if start_step is not None else self.start_step
            self.end_step = end_step if end_step is not None else self.end_step
            
            # 重新初始化数据结构
            self.power_data = {
                'qkv_power': defaultdict(list),       # QKV计算功耗
                'proj_power': defaultdict(list),      # 投影计算功耗
                'fc1_power': defaultdict(list),       # MLP第一层功耗
                'fc2_power': defaultdict(list),       # MLP第二层功耗
                'dequant_power': defaultdict(list), # 反量化的功耗
                'layer_power': defaultdict(list),     # 每层总功耗
                'qkv_sensing': defaultdict(list),     # QKV的sensing功耗
                'proj_sensing': defaultdict(list),    # 投影的sensing功耗
                'fc1_sensing': defaultdict(list),     # MLP第一层的sensing功耗
                'fc2_sensing': defaultdict(list),     # MLP第二层的sensing功耗
                'dequant_sensing': defaultdict(list)   # 反量化的sensing功耗
            }
            self.total_power = 0
            self.total_sensing_power = 0
            self.layer_count = 0
            self.power_per_layer = {}
            
            # 重新解析日志文件
            self._parse_log_file()
            
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 构建步骤范围字符串，用于生成对应的文件名
        step_range_str = ""
        if self.start_step is not None and self.end_step is not None:
            step_range_str = f"_step{self.start_step}-{self.end_step}"
        elif self.start_step is not None:
            step_range_str = f"_step{self.start_step}plus"
        elif self.end_step is not None:
            step_range_str = f"_step0-{self.end_step}"
        
        # 生成各类图表，文件名中添加步骤范围信息
        self.plot_power_by_operation(os.path.join(output_dir, f'operation_power_distribution{step_range_str}.png'), save_data)
        self.plot_sensing_power_by_operation(os.path.join(output_dir, f'sensing_power_distribution{step_range_str}.png'), save_data)
        self.plot_power_by_layer(os.path.join(output_dir, f'layer_power_distribution{step_range_str}.png'), save_data)
        self.plot_sensing_vs_total(os.path.join(output_dir, f'sensing_vs_total_power{step_range_str}.png'), save_data)
        
        # 步骤功耗分析
        self.plot_step_power(
            os.path.join(output_dir, f'step_power_distribution{step_range_str}.png'), 
            save_data
        )
        
        # 如果没有指定层，则分析前几层和后几层
        if detail_layers is None:
            detail_layers = [0, 1, self.layer_count//2, self.layer_count-2, self.layer_count-1] if self.layer_count > 0 else []
        
        # 分析指定的层
        for layer in detail_layers:
            if layer < self.layer_count:
                self.plot_layer_composition(layer, os.path.join(output_dir, f'layer_{layer}_power_composition{step_range_str}.png'), save_data)
        
        # 生成组合报告
        if save_data:
            self.generate_combined_report(output_dir)
        
        # 打印摘要
        self.print_power_summary()
        
        return output_dir


# 创建GUI界面
class PowerAnalysisGUI:
    """提供用户友好的图形界面来分析GPT-2模型的功耗表现"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("GPT-2 模型功耗分析工具")
        self.root.geometry("1000x700")
        
        # 添加一个标志变量来控制分析进程的退出
        self.running = False
        
        # 当窗口关闭时，确保所有线程也被终止
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 设置风格
        style = ttk.Style()
        style.configure('TButton', font=('Arial', 10))
        style.configure('TLabel', font=('Arial', 10))
        
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding=10)
        control_frame.pack(fill=tk.X, pady=5, padx=5)
        
        # 日志文件选择
        ttk.Label(control_frame, text="日志文件:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.file_entry = ttk.Entry(control_frame, width=50)
        self.file_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        ttk.Button(control_frame, text="浏览...", command=self.browse_file).grid(row=0, column=2, padx=5, pady=5)
        
        # 输出目录选择
        ttk.Label(control_frame, text="输出目录:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.output_dir_entry = ttk.Entry(control_frame, width=50)
        self.output_dir_entry.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
        ttk.Button(control_frame, text="浏览...", command=self.browse_output_dir).grid(row=1, column=2, padx=5, pady=5)
        
        # 图表类型选择
        ttk.Label(control_frame, text="图表类型:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.chart_type = tk.StringVar(value="all")
        chart_frame = ttk.Frame(control_frame)
        chart_frame.grid(row=2, column=1, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        ttk.Radiobutton(chart_frame, text="所有图表", variable=self.chart_type, value="all").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(chart_frame, text="操作功耗分布", variable=self.chart_type, value="operation").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(chart_frame, text="Sensing功耗分布", variable=self.chart_type, value="sensing").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(chart_frame, text="层功耗分布", variable=self.chart_type, value="layer").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(chart_frame, text="步骤功耗分布", variable=self.chart_type, value="step").pack(side=tk.LEFT, padx=5)
        
        # 层选择
        ttk.Label(control_frame, text="分析特定层(逗号分隔):").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.layer_entry = ttk.Entry(control_frame, width=50)
        self.layer_entry.grid(row=3, column=1, sticky=tk.EW, padx=5, pady=5)
        ttk.Label(control_frame, text="分析步骤范围:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        step_range_frame = ttk.Frame(control_frame)
        step_range_frame.grid(row=4, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(step_range_frame, text="从:").pack(side=tk.LEFT, padx=2)
        self.step_start = ttk.Entry(step_range_frame, width=5)
        self.step_start.pack(side=tk.LEFT, padx=2)
        ttk.Label(step_range_frame, text="到:").pack(side=tk.LEFT, padx=2)
        self.step_end = ttk.Entry(step_range_frame, width=5)
        self.step_end.pack(side=tk.LEFT, padx=2)
        ttk.Label(step_range_frame, text="(留空表示所有步骤)").pack(side=tk.LEFT, padx=5)       
        # 保存数据选项
        self.save_data = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="保存图表数据为CSV/Excel", variable=self.save_data).grid(
            row=3, column=2, sticky=tk.W, padx=5, pady=5
        )
        
        # 添加进度条
        ttk.Label(control_frame, text="分析进度:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
        self.progress = ttk.Progressbar(control_frame, orient="horizontal", length=400, mode="indeterminate")
        self.progress.grid(row=5, column=1, sticky=tk.EW, padx=5, pady=5)
        
        # 按钮
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=6, column=0, columnspan=3, pady=10)
        self.analyze_button = ttk.Button(button_frame, text="分析", command=self.analyze)
        self.analyze_button.pack(side=tk.LEFT, padx=10)
        self.export_button = ttk.Button(button_frame, text="导出报告", command=self.export_report)
        self.export_button.pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="退出", command=self.on_closing).pack(side=tk.LEFT, padx=10)
        
        # 创建日志区域
        log_frame = ttk.LabelFrame(main_frame, text="分析日志", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)
        
        self.log_text = tk.Text(log_frame, wrap=tk.WORD, height=15)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建滚动条
        scrollbar = ttk.Scrollbar(self.log_text, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)
        
        # 创建结果显示区域
        result_frame = ttk.LabelFrame(main_frame, text="分析结果", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)
        
        self.result_text = tk.Text(result_frame, wrap=tk.WORD, height=15)
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建滚动条
        result_scrollbar = ttk.Scrollbar(self.result_text, command=self.result_text.yview)
        result_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.config(yscrollcommand=result_scrollbar.set)
        
        # 初始化分析器和结果
        self.analyzer = None
        self.result_dir = None
        
        # 默认输出目录
        default_output = os.path.join(os.getcwd(), "power_analysis_results")
        self.output_dir_entry.insert(0, default_output)
        
        # 结果锁，防止多线程访问冲突
        self.result_lock = threading.Lock()
        
        # 重定向输出到日志窗口
        sys.stdout = self
        
        # 添加信息到日志窗口
        self.log_text.insert(tk.END, "欢迎使用GPT-2模型功耗分析工具!\n")
        self.log_text.insert(tk.END, "请选择日志文件并点击'分析'按钮开始分析。\n")
        
    def on_closing(self):
        """窗口关闭时的处理"""
        if self.running:
            if messagebox.askokcancel("退出", "分析正在进行中，确定要退出吗?"):
                self.running = False  # 停止分析线程
                time.sleep(0.5)  # 给线程一点时间来退出
                self.root.destroy()
        else:
            self.root.destroy()
        
    def write(self, text):
        """重定向输出到日志窗口"""
        # 使用after方法确保UI更新是安全的
        self.root.after(0, self._update_log, text)
        
    def _update_log(self, text):
        """实际更新日志窗口的方法，在主线程中执行"""
        try:
            self.log_text.insert(tk.END, text)
            self.log_text.see(tk.END)
            self.log_text.update_idletasks()
        except tk.TclError:
            # 如果窗口已关闭，忽略错误
            pass
        
    def flush(self):
        """为sys.stdout实现的flush方法"""
        pass
        
    def browse_file(self):
        """浏览并选择日志文件"""
        file_path = filedialog.askopenfilename(
            title="选择GPT-2日志文件",
            filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, file_path)
    
    def browse_output_dir(self):
        """浏览并选择输出目录"""
        dir_path = filedialog.askdirectory(title="选择输出目录")
        if dir_path:
            self.output_dir_entry.delete(0, tk.END)
            self.output_dir_entry.insert(0, dir_path)
    
 # 在PowerAnalysisGUI类中替换analyze方法
    def analyze(self):
        """执行分析，使用更简单的线程模型来避免GUI闪退"""
        # 检查是否已经在运行分析
        if self.running:
            messagebox.showwarning("警告", "分析已经在进行中")
            return
            
        log_file = self.file_entry.get().strip()
        output_dir = self.output_dir_entry.get().strip()
        chart_type = self.chart_type.get()
        save_data = self.save_data.get()
        
        if not log_file:
            messagebox.showerror("错误", "请选择日志文件")
            return
            
        if not os.path.isfile(log_file):
            messagebox.showerror("错误", f"文件不存在: {log_file}")
            return
            
        # 清空日志和结果区域
        self.log_text.delete(1.0, tk.END)
        self.result_text.delete(1.0, tk.END)
        
        # 创建输出目录
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            messagebox.showerror("错误", f"无法创建输出目录: {str(e)}")
            return
            
        # 解析层选择
        detail_layers = None
        if self.layer_entry.get().strip():
            try:
                layer_str = self.layer_entry.get().strip()
                detail_layers = [int(l.strip()) for l in layer_str.split(',')]
            except:
                messagebox.showerror("错误", "层指定格式错误，请使用逗号分隔的整数")
                return
        
        # 解析步骤范围
        start_step = None
        end_step = None
        
        if self.step_start.get().strip():
            try:
                start_step = int(self.step_start.get().strip())
            except:
                messagebox.showerror("错误", "起始步骤必须是整数")
                return
        
        if self.step_end.get().strip():
            try:
                end_step = int(self.step_end.get().strip())
            except:
                messagebox.showerror("错误", "结束步骤必须是整数")
                return
        
        # 检查步骤范围的有效性
        if start_step is not None and end_step is not None and start_step > end_step:
            messagebox.showerror("错误", "起始步骤不能大于结束步骤")
            return
        
        # 添加步骤范围信息到文件名
        step_range_str = ""
        if start_step is not None and end_step is not None:
            step_range_str = f"_step{start_step}-{end_step}"
        elif start_step is not None:
            step_range_str = f"_step{start_step}plus"
        elif end_step is not None:
            step_range_str = f"_step0-{end_step}"
            
        # 设置UI元素的状态
        self.analyze_button.config(state=tk.DISABLED)
        self.export_button.config(state=tk.DISABLED)
        self.progress.start(10)  # 开始进度条动画
        
        # 标记分析开始
        self.running = True
        
        # 为了避免作用域混淆，存储输出目录
        final_output_dir = output_dir
        
        # 定义单独的分析函数，稍后会在线程中调用
        def do_analysis():
            try:
                # 创建分析器对象，传入步骤范围
                self.write(f"开始分析日志文件: {log_file}\n")
                if start_step is not None or end_step is not None:
                    self.write(f"分析步骤范围: {start_step if start_step is not None else '开始'} 到 {end_step if end_step is not None else '结束'}\n")
                    
                try:
                    # 关键修改: 创建分析器时传入步骤范围
                    self.analyzer = GPT2PowerAnalyzer(log_file, start_step, end_step)
                except Exception as e:
                    self.write(f"\n日志文件解析失败: {str(e)}\n")
                    raise e
                
                # 检查线程是否应该终止
                if not self.running:
                    return False
                    
                # 分析器已创建，进行实际分析
                if chart_type == "all":
                    # 执行全面分析
                    result_dir = self.analyzer.analyze_all(final_output_dir, detail_layers, save_data)
                    self.write(f"分析完成，结果保存到: {result_dir}\n")
                    # 显示生成的第一个图表
                    if os.path.exists(os.path.join(result_dir, f'operation_power_distribution{step_range_str}.png')):
                        self.display_image(os.path.join(result_dir, f'operation_power_distribution{step_range_str}.png'))
                    
                elif chart_type == "operation":
                    # 按操作类型分析
                    chart_path = self.analyzer.plot_power_by_operation(os.path.join(final_output_dir, f'operation_power{step_range_str}.png'), save_data)
                    self.write(f"按操作类型的功耗分析完成，图表保存为: {chart_path}\n")
                    self.display_image(chart_path)
                    
                elif chart_type == "sensing":
                    # Sensing功耗分析
                    chart_path = self.analyzer.plot_sensing_power_by_operation(os.path.join(final_output_dir, f'sensing_power{step_range_str}.png'), save_data)
                    self.write(f"Sensing功耗分析完成，图表保存为: {chart_path}\n")
                    self.display_image(chart_path)
                    
                elif chart_type == "layer":
                    # 按层分析
                    chart_path = self.analyzer.plot_power_by_layer(os.path.join(final_output_dir, f'layer_power{step_range_str}.png'), save_data)
                    self.write(f"按层的功耗分析完成，图表保存为: {chart_path}\n")
                    self.display_image(chart_path)
                    
                elif chart_type == "sensing_vs_total":
                    # Sensing与总功耗比较
                    chart_path = self.analyzer.plot_sensing_vs_total(os.path.join(final_output_dir, f'sensing_vs_total{step_range_str}.png'), save_data)
                    self.write(f"Sensing与总功耗比较分析完成，图表保存为: {chart_path}\n")
                    self.display_image(chart_path)
                    
                elif chart_type == "step":
                    # 步骤分析
                    chart_path = self.analyzer.plot_step_power(os.path.join(final_output_dir, f'step_power{step_range_str}.png'), save_data)
                    if chart_path:
                        self.write(f"步骤功耗分析完成，图表保存为: {chart_path}\n")
                        self.display_image(chart_path)
                    else:
                        self.write("步骤功耗分析无结果，可能是日志中没有步骤信息\n")
                    
                elif chart_type == "layer_detail":
                    # 层详情分析 - 需要指定层
                    if detail_layers is None or len(detail_layers) == 0:
                        self.write("错误: 需要指定要分析的层编号\n")
                    else:
                        for layer in detail_layers:
                            chart_path = self.analyzer.plot_layer_composition(
                                layer, 
                                os.path.join(final_output_dir, f'layer_{layer}_detail{step_range_str}.png'), 
                                save_data
                            )
                            self.write(f"第 {layer} 层功耗组成分析完成\n")
                            # 显示第一个层的图表
                            if layer == detail_layers[0]:
                                self.display_image(chart_path)
                
                # 打印功耗摘要
                self.analyzer.print_power_summary()
                
            except Exception as e:
                self.write(f"\n分析失败: {str(e)}\n")
                import traceback
                self.write(traceback.format_exc())
            
            # 恢复UI元素的状态
            self.analyze_button.config(state=tk.NORMAL)
            self.export_button.config(state=tk.NORMAL)
            self.progress.stop()  # 停止进度条动画
            self.running = False  # 标记分析结束
        
        # 创建并启动分析线程
        thread = threading.Thread(target=do_analysis)
        thread.daemon = True
        thread.start()
    # 添加新方法来处理分析结果
    def display_image(self, image_path):
        """在结果区域显示图像"""
        try:
            # 清空结果文本区域
            self.result_text.delete(1.0, tk.END)
            
            # 使用PIL打开图像
            from PIL import Image, ImageTk
            img = Image.open(image_path)
            
            # 调整图像大小以适应显示区域
            display_width = self.result_text.winfo_width() - 20
            width, height = img.size
            ratio = height / width
            
            new_width = min(display_width, width)
            new_height = int(new_width * ratio)
            
            img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # 转换为PhotoImage
            photo = ImageTk.PhotoImage(img)
            
            # 在文本区域中显示图像
            self.result_text.image = photo  # 保留引用，防止被垃圾回收
            self.result_text.insert(tk.END, "\n")
            self.result_text.image_create(tk.END, image=photo)
            self.result_text.insert(tk.END, f"\n\n图像已保存到: {image_path}")
            
        except Exception as e:
            self.write(f"\n无法显示图像: {str(e)}\n")
            self.result_text.insert(tk.END, f"图像已保存到: {image_path}\n但无法在界面中显示。")
    def process_analysis_result(self, result, output_dir):
        """在主线程中处理分析结果"""
        # 停止进度指示和重置运行状态
        self.progress.stop()
        self.running = False
        
        # 恢复按钮状态
        self.analyze_button.config(state=tk.NORMAL)
        self.export_button.config(state=tk.NORMAL)
        
        if result == False:
            # 分析被取消
            self.write("\n分析已取消\n")
            return
        
        if isinstance(result, str):
            if result.startswith("分析过程中出现错误"):
                # 处理错误
                messagebox.showerror("错误", result)
                return
                
            # 处理成功完成的分析
            self.write("\n分析完成!\n")
            
            if result == "all":
                self.show_results()
            elif result == "operation":
                self.show_result("操作功耗分布", os.path.join(output_dir, 'operation_power_distribution.png'))
            elif result == "sensing":
                self.show_result("Sensing功耗分布", os.path.join(output_dir, 'sensing_power_distribution.png'))
            elif result == "layer":
                self.show_result("层功耗分布", os.path.join(output_dir, 'layer_power_distribution.png'))
            elif result == "step":
                # 使用保存的步骤结果文件名
                file_name = getattr(self, 'step_result_file', 'step_power_distribution.png')
                self.show_result("步骤功耗分布", os.path.join(output_dir, file_name))
                
            # 显示成功信息，包括数据文件信息
            if self.save_data.get():
                messagebox.showinfo("成功", f"分析完成!\n\n结果图表已保存到: {output_dir}\n\n图表数据已保存为CSV和Excel文件")
            else:
                messagebox.showinfo("成功", f"分析完成!\n\n结果图表已保存到: {output_dir}")
            
    def finish_analysis(self, output_dir):
        """完成分析后的处理"""
        # 停止进度条
        self.progress.stop()
        # 恢复按钮状态
        self.analyze_button.config(state=tk.NORMAL)
        self.export_button.config(state=tk.NORMAL)
        # 显示成功消息
        if self.save_data.get():
            messagebox.showinfo("成功", f"分析完成!\n\n结果图表已保存到: {output_dir}\n\n图表数据已保存为CSV和Excel文件")
        else:
            messagebox.showinfo("成功", f"分析完成!\n\n结果图表已保存到: {output_dir}")
        
    def handle_error(self, error_msg):
        """处理分析过程中的错误"""
        # 停止进度条
        self.progress.stop()
        # 恢复按钮状态
        self.analyze_button.config(state=tk.NORMAL)
        self.export_button.config(state=tk.NORMAL)
        # 显示错误消息
        messagebox.showerror("错误", error_msg)
    
    def show_results(self):
        """显示分析结果摘要"""
        if not self.analyzer:
            return
            
        # 获取功率数据
        operation_power = self.analyzer.analyze_by_operation()
        sensing_power = self.analyzer.analyze_sensing_by_operation()
        
        # 格式化结果
        result = "===== 分析结果摘要 =====\n\n"
        result += f"总功耗: {self.analyzer.total_power:.2f} uJ\n"
        result += f"总Sensing功耗: {self.analyzer.total_sensing_power:.2f} uJ\n"
        result += f"Sensing占比: {(self.analyzer.total_sensing_power/self.analyzer.total_power*100):.2f}%\n\n"
        
        result += "操作功耗分布:\n"
        for op, power in sorted(operation_power.items(), key=lambda x: x[1], reverse=True):
            result += f"- {op}: {power:.2f} uJ\n"
            
        result += "\nSensing功耗分布:\n"
        for op, power in sorted(sensing_power.items(), key=lambda x: x[1], reverse=True):
            result += f"- {op}: {power:.2f} uJ\n"
        
        result += f"\n结果图表已保存到: {self.result_dir}\n"
        
        if self.save_data.get():
            result += "\n图表数据已保存为CSV和Excel文件，可用于后续分析或可视化。"
        
        # 显示结果
        with self.result_lock:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, result)
    
    def show_result(self, title, file_path):
        """显示单个分析结果"""
        result = f"===== {title} =====\n\n"
        result += f"结果已保存到: {file_path}\n"
        
        if self.save_data.get():
            data_csv = os.path.splitext(file_path)[0] + "_data.csv"
            data_excel = os.path.splitext(file_path)[0] + "_data.xlsx"
            result += f"\n图表数据已保存为:\n- CSV: {data_csv}\n- Excel: {data_excel}"
        
        # 显示结果
        with self.result_lock:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, result)
    
    def export_report(self):
        """导出分析报告"""
        if not self.analyzer:
            messagebox.showwarning("警告", "请先进行分析")
            return
            
        # 询问报告类型
        report_type = messagebox.askyesno("导出报告", "你想要导出完整Excel报告吗?\n\n选择'是'导出含有多个工作表的Excel报告\n选择'否'导出基础TXT报告")
        
        if report_type:  # 导出Excel报告
            save_path = filedialog.asksaveasfilename(
                title="保存分析报告",
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
            )
            
            if not save_path:
                return
                
            try:
                # 创建一个目录用于临时存储报告
                temp_dir = os.path.join(os.path.dirname(save_path), "temp_report")
                os.makedirs(temp_dir, exist_ok=True)
                
                # 生成报告
                self.analyzer.generate_combined_report(os.path.dirname(save_path))
                
                messagebox.showinfo("成功", f"Excel分析报告已保存到: {save_path}")
            except Exception as e:
                messagebox.showerror("错误", f"保存Excel报告时出错: {str(e)}")
        else:  # 导出TXT报告
            save_path = filedialog.asksaveasfilename(
                title="保存分析报告",
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            
            if not save_path:
                return
                
            try:
                with open(save_path, 'w', encoding='gbk') as f:
                    # 功率数据
                    operation_power = self.analyzer.analyze_by_operation()
                    sensing_power = self.analyzer.analyze_sensing_by_operation()
                    
                    f.write("===== GPT-2 模型功耗分析报告 =====\n\n")
                    f.write(f"分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"日志文件: {self.analyzer.log_file}\n\n")
                    
                    f.write("== 总体统计 ==\n")
                    f.write(f"总功耗: {self.analyzer.total_power:.2f} uJ\n")
                    f.write(f"总Sensing功耗: {self.analyzer.total_sensing_power:.2f} uJ\n")
                    f.write(f"Sensing占比: {(self.analyzer.total_sensing_power/self.analyzer.total_power*100):.2f}%\n")
                    f.write(f"层数: {self.analyzer.layer_count}\n\n")
                    
                    f.write("== 操作功耗分布 ==\n")
                    for op, power in sorted(operation_power.items(), key=lambda x: x[1], reverse=True):
                        total = sum(operation_power.values())
                        f.write(f"{op}: {power:.2f} uJ ({power/total*100:.1f}%)\n")
                    
                    f.write("\n== Sensing功耗分布 ==\n")
                    for op, power in sorted(sensing_power.items(), key=lambda x: x[1], reverse=True):
                        total = sum(sensing_power.values())
                        f.write(f"{op}: {power:.2f} uJ ({power/total*100:.1f}%)\n")
                    
                    f.write("\n== 层功耗分布 ==\n")
                    layer_power = self.analyzer.analyze_by_layer()
                    for layer, power in sorted(layer_power.items(), key=lambda x: x[1], reverse=True):
                        f.write(f"{layer}: {power:.2f} uJ\n")
                    
                    f.write("\n== 硬件效率分析 ==\n")
                    f.write(f"每层平均功耗: {sum(operation_power.values())/self.analyzer.layer_count:.2f} uJ\n")
                    f.write(f"功耗最高层: 层 {max(self.analyzer.power_per_layer.items(), key=lambda x: x[1])[0] if self.analyzer.power_per_layer else 'N/A'}\n")
                    f.write(f"功耗最低层: 层 {min(self.analyzer.power_per_layer.items(), key=lambda x: x[1])[0] if self.analyzer.power_per_layer else 'N/A'}\n")
                
                messagebox.showinfo("成功", f"分析报告已保存到: {save_path}")
            except Exception as e:
                messagebox.showerror("错误", f"保存报告时出错: {str(e)}")


# 主程序入口
if __name__ == "__main__":
    # 添加必要的模块导入
    import time
    
    # GUI模式
    root = tk.Tk()
    app = PowerAnalysisGUI(root)
    root.mainloop()
