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
import torch
import config
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import re
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
class GPT2TimeAnalyzer:
    """
    GPT-2模型时间性能分析器
    用于解析日志文件并分析硬件加速时间消耗
    """
    
    def __init__(self, log_file):
        """
        初始化分析器
        
        参数:
        - log_file: 日志文件路径
        """
        self.log_file = log_file
        self.time_data = { 
            'qkv_times': defaultdict(list),       # QKV计算时间
            'proj_times': defaultdict(list),      # 投影计算时间
            'fc1_times': defaultdict(list),       # MLP第一层时间
            'fc2_times': defaultdict(list),       # MLP第二层时间
            'dequant_times': defaultdict(list),   # 反量化时间
            'layer_times': defaultdict(list),     # 每层总时间
        }
        self.write_time = 0                      # 权重加载时间
        self.total_hardware_time = 0             # 总硬件计算时间
        self.total_generation_time = 0           # 总生成时间
        self.layer_count = 0                     # 层数
        self.time_per_layer = {}                 # 每层时间统计
        
        # 操作类型映射表（中文到英文）
        self.operation_map = {
            'QKV计算': 'QKV Computation',
            '注意力投影': 'Attention Projection',
            'MLP第一层': 'MLP First Layer',
            'MLP第二层': 'MLP Second Layer',
            '反量化': 'Dequantization',
            '总时间': 'Total Time'
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
        """解析日志文件，提取时间信息"""
        print(f"Parsing log file: {self.log_file}")
        
        # 定义正则表达式模式
        compute_time_pattern = re.compile(r'计算(.*?)，耗时(\d+\.?\d*)us，目前总耗时(\d+\.?\d*)us')
        layer_start_pattern = re.compile(r'开始处理第(\d+)层')
        generation_time_pattern = re.compile(r'文本生成完成，耗时：(\d+\.?\d*)秒，硬件计算耗时：(\d+\.?\d*)毫秒')
        
        current_layer = -1
        
        with open(self.log_file, 'r', encoding='gbk') as f:
            for line in f:
                # 检查层开始
                layer_match = layer_start_pattern.search(line)
                if layer_match:
                    current_layer = int(layer_match.group(1))
                    self.layer_count = max(self.layer_count, current_layer + 1)
                    continue
                
                compute_match = compute_time_pattern.search(line)
                if compute_match:
                    operation = compute_match.group(1)
                    time_cost = float(compute_match.group(2))
                    if "反量化" in operation:
                        self.time_data['dequant_times'][current_layer].append(time_cost)
                    elif "QKV" in operation:
                        self.time_data['qkv_times'][current_layer].append(time_cost)
                    elif "注意力输出投影" in operation:
                        self.time_data['proj_times'][current_layer].append(time_cost)
                    elif "MLP第一层" in operation:
                        self.time_data['fc1_times'][current_layer].append(time_cost)
                    elif "MLP第二层" in operation:
                        self.time_data['fc2_times'][current_layer].append(time_cost)
                    
                    continue
                
                # 检查总时间
                generation_match = generation_time_pattern.search(line)
                if generation_match:
                    self.total_generation_time = float(generation_match.group(1))
                    self.total_hardware_time = float(generation_match.group(2))
        
        # 计算每层总时间
        for layer in range(self.layer_count):
            layer_total = 0
            for op_type in [ 'qkv_times', 'proj_times', 'fc1_times', 'fc2_times', 'dequant_times']:
                if layer in self.time_data[op_type]:
                    layer_total += sum(self.time_data[op_type][layer])
            self.time_data['layer_times'][layer] = [layer_total]
            self.time_per_layer[layer] = layer_total
            
        print(f"Parsing complete, found {self.layer_count} layers")

    def analyze_by_operation(self):
        """按操作类型分析时间消耗"""
        operation_times = {
            'QKV计算': sum(sum(times) for times in self.time_data['qkv_times'].values()),
            '注意力投影': sum(sum(times) for times in self.time_data['proj_times'].values()),
            'MLP第一层': sum(sum(times) for times in self.time_data['fc1_times'].values()),
            'MLP第二层': sum(sum(times) for times in self.time_data['fc2_times'].values()),
            '反量化': sum(sum(times) for times in self.time_data['dequant_times'].values()),
        }
        
        # 将微秒转换为毫秒以便于阅读
        operation_times = {k: v/1000 for k, v in operation_times.items()}
        
        return operation_times
    
    def analyze_by_layer(self):
        """按层分析时间消耗"""
        layer_times = {}
        for layer in range(self.layer_count):
            if layer in self.time_data['layer_times']:
                # 从微秒转换为毫秒
                layer_times[f"层 {layer}"] = self.time_data['layer_times'][layer][0] / 1000
        
        return layer_times
    
    def analyze_layer_composition(self, layer_num):
        """分析指定层的时间组成"""
        if layer_num not in range(self.layer_count):
            print(f"Error: Layer {layer_num} does not exist")
            return {}
            
        layer_composition = {
            'QKV计算': sum(self.time_data['qkv_times'].get(layer_num, [])),
            '注意力投影': sum(self.time_data['proj_times'].get(layer_num, [])),
            'MLP第一层': sum(self.time_data['fc1_times'].get(layer_num, [])),
            'MLP第二层': sum(self.time_data['fc2_times'].get(layer_num, [])),
            '反量化': sum(self.time_data['dequant_times'].get(layer_num, []))
        }
        
        # 将微秒转换为毫秒
        layer_composition = {k: v/1000 for k, v in layer_composition.items()}
        
        return layer_composition

    def plot_time_by_operation(self, save_path='operation_time_distribution.png'):
        """使用增强型饼图创建3D效果的操作时间分布图"""
        operation_times = self.analyze_by_operation()
        
        # 创建英文键字典
        eng_operation_times = {}
        for k, v in operation_times.items():
            eng_operation_times[self.operation_map.get(k, k)] = v
        
        # 计算每个扇区的偏移量，使数值大的扇区凸出
        values = list(eng_operation_times.values())
        labels = list(eng_operation_times.keys())
        total = sum(values)
        explode = [0.05 + 0.05 * (v / max(values)) for v in values]
        
        # 准备包含百分比和时间的标签
        labels_with_values = []
        for i, label in enumerate(labels):
            pct = values[i] / total * 100
            labels_with_values.append(f'{label}\n{pct:.1f}% ({values[i]:.1f} ms)')
        
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
        
        # 添加图例，包含标签、百分比和时间
        plt.legend(
            wedges, 
            labels_with_values,
            title="Operation Types",
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            fontsize=10
        )
        plt.axis('equal')  # 确保饼图是圆形的
        
        # 保存图表
        plt.tight_layout()
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"Operation time distribution 3D-style pie chart saved as '{save_path}'")
    
    def plot_time_by_layer(self, save_path='layer_time_distribution.png'):
        """Plot time consumption by layer as a bar chart"""
        layer_times = self.analyze_by_layer()
        
        # 翻译层名称
        eng_layer_times = {}
        for k, v in layer_times.items():
            layer_num = k.split(' ')[1]  # 提取数字部分
            eng_layer_times[f"Layer {layer_num}"] = v
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(eng_layer_times)), eng_layer_times.values(),color='orange')
        plt.xlabel('Layer Index')
        plt.ylabel('Time (ms)')
        plt.title('Computation Time per Layer in GPT-2 Model')
        plt.xticks(range(len(eng_layer_times)), eng_layer_times.keys())
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    '%.1f' % height,
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"Layer time distribution bar chart saved as '{save_path}'")
    
    def plot_layer_composition(self, layer_num, save_path=None):
        """使用增强型饼图创建3D效果的层时间组成图"""
        if save_path is None:
            save_path = f'layer_{layer_num}_composition.png'
            
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
            print(f"No data to display for Layer {layer_num}")
            return
            
        total = sum(values)
        explode = [0.05 + 0.05 * (v / max(values)) for v in values]
        
        # 准备包含百分比和时间的标签
        labels_with_values = []
        for i, label in enumerate(labels):
            pct = values[i] / total * 100
            labels_with_values.append(f'{label}\n{pct:.1f}% ({values[i]:.1f} ms)')
        
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
        
        # 添加图例，包含标签、百分比和时间
        plt.legend(
            wedges, 
            labels_with_values,
            title=f"Layer {layer_num} Operations",
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            fontsize=10
        )
            
        plt.axis('equal')  # 确保饼图是圆形的
        
        # 保存图表
        plt.tight_layout()
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"Layer {layer_num} time composition 3D-style pie chart saved as '{save_path}'")
        
    def print_time_summary(self):
        """打印时间消耗摘要"""
        operation_times = self.analyze_by_operation()
        total_op_time = sum(operation_times.values())
        
        print("\n===== GPT-2 Model Time Performance Analysis =====")
        print(f"Total Hardware Computation Time: {self.total_hardware_time:.2f} millisecond")
        print(f"Total Text Generation Time: {self.total_generation_time:.2f} seconds")
        print(f"Hardware Acceleration Ratio: {(self.total_hardware_time/self.total_generation_time*0.1):.3f}%")
        print("\n----- Operation Type Time Distribution -----")
        for op, time in sorted(operation_times.items(), key=lambda x: x[1], reverse=True):
            eng_op = self.operation_map.get(op, op)
            print(f"{eng_op}: {time:.2f} ms ({time/total_op_time*100:.1f}%)")
            
        print("\n----- Layer Time Distribution TOP5 -----")
        layer_times = self.analyze_by_layer()
        for layer, time in sorted(layer_times.items(), key=lambda x: x[1], reverse=True)[:5]:
            layer_num = layer.split(' ')[1]
            print(f"Layer {layer_num}: {time:.2f} ms")
            
        print("\n----- Hardware Efficiency Analysis -----")
        print(f"Average computation time per layer: {total_op_time/self.layer_count:.2f} ms")
        print(f"Slowest layer: Layer {max(self.time_per_layer.items(), key=lambda x: x[1])[0]}")
        print(f"Fastest layer: Layer {min(self.time_per_layer.items(), key=lambda x: x[1])[0]}")
        print("============================")
    
    def analyze_all(self, output_dir="results", detail_layers=None):
        """执行完整分析并生成所有图表
        
        参数:
        - output_dir: 输出图像的目录
        - detail_layers: 要详细分析的特定层列表
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成各类图表
        self.plot_time_by_operation(os.path.join(output_dir, 'operation_time_distribution.png'))
        self.plot_time_by_layer(os.path.join(output_dir, 'layer_time_distribution.png'))
        
        # 如果指定了详细层，则对这些层进行详细分析
        if detail_layers:
            for layer in detail_layers:
                if layer in range(self.layer_count):
                    self.plot_layer_composition(layer, os.path.join(output_dir, f'layer_{layer}_composition.png'))
        else:
            # 默认分析第0层和最后一层
            self.plot_layer_composition(0, os.path.join(output_dir, 'layer_0_composition.png'))
            if self.layer_count > 1:
                self.plot_layer_composition(self.layer_count - 1, 
                                          os.path.join(output_dir, f'layer_{self.layer_count-1}_composition.png'))
            
        self.print_time_summary()
        
        # 保存分析结果为CSV
        self._save_analysis_to_csv(output_dir)
        
        return {
            "total_hardware_time": self.total_hardware_time,
            "total_generation_time": self.total_generation_time,
            "operations": self.analyze_by_operation(),
            "layers": self.analyze_by_layer()
        }
    
    def _save_analysis_to_csv(self, output_dir):
        """将分析结果保存为CSV文件"""
        # 保存操作时间分布
        operation_times = self.analyze_by_operation()
        ops_df = pd.DataFrame({
            'Operation': [self.operation_map.get(op, op) for op in operation_times.keys()],
            'Time (ms)': list(operation_times.values())
        })
        ops_df.to_csv(os.path.join(output_dir, 'operation_times.csv'), index=False, encoding='utf-8-sig')
        
        # 保存层时间分布
        layer_times = self.analyze_by_layer()
        layers_df = pd.DataFrame({
            'Layer': [f"Layer {k.split(' ')[1]}" for k in layer_times.keys()],
            'Time (ms)': list(layer_times.values())
        })
        layers_df.to_csv(os.path.join(output_dir, 'layer_times.csv'), index=False, encoding='utf-8-sig')
        
        # 保存每层各操作时间详情
        operations = ['QKV计算', '注意力投影', 'MLP第一层', 'MLP第二层', '反量化']
        eng_operations = [self.operation_map[op] for op in operations]
        
        layer_op_data = []
        for layer in range(self.layer_count):
            layer_comp = self.analyze_layer_composition(layer)
            row_data = {'Layer': f"Layer {layer}"}
            for i, op in enumerate(operations):
                row_data[eng_operations[i]] = layer_comp.get(op, 0)
            layer_op_data.append(row_data)
        
        layer_op_df = pd.DataFrame(layer_op_data)
        layer_op_df.to_csv(os.path.join(output_dir, 'layer_operation_times.csv'), index=False, encoding='utf-8-sig')
        
        print(f"Analysis results saved to CSV files in '{output_dir}' directory")

class TimeAnalyzerGUI:
    """交互式GPT-2模型时间分析工具，支持比较任意数量的日志文件"""
    
    def __init__(self, root):
        """初始化GUI界面"""
        self.root = root
        self.root.title("GPT-2 Time Analyzer")
        self.root.geometry("1000x700")
        
        self.log_files = []  # 存储所有日志文件路径
        self.log_entries = []  # 存储所有日志文件输入框和相关部件
        self.log_labels = []  # 存储日志文件的自定义标签
        
        self.create_widgets()
        
        # 初始添加两个日志文件输入
        self.add_log_file()
        self.add_log_file()
        
        # 添加：窗口调整大小事件处理
        self.root.bind("<Configure>", self.on_window_resize)
        # 添加：最大化按钮
        maximize_btn = ttk.Button(self.main_frame, text="Maximize Window", 
                                command=lambda: self.root.state('zoomed'))
        maximize_btn.grid(row=0, column=2, sticky="e", padx=5, pady=5)
    def on_window_resize(self, event):
        """处理窗口大小变化事件"""
        # 仅当窗口本身大小变化时才处理，避免子组件事件
        if event.widget == self.root:
            # 重新调整日志区域的高度
            new_height = max(200, event.height - 400)  # 给其他区域留出空间
            self.canvas.config(height=new_height)
            # 更新滚动区域
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    def create_widgets(self):
        """创建GUI组件"""
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # 主框架配置
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 为主框架的行设置权重，使其能正确分配空间
        for i in range(7):  # 总共7行
            self.main_frame.rowconfigure(i, weight=1 if i == 6 else 0)  # 结果区域行权重为1
        for i in range(3):  # 3列
            self.main_frame.columnconfigure(i, weight=1)

        
        # 创建标题标签
        title_label = ttk.Label(self.main_frame, text="GPT-2 Model Time Analyzer", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=10)
        
        # 创建用于存放日志文件输入框的容器
        self.logs_container = ttk.Frame(self.main_frame)
        self.logs_container.grid(row=1, column=0, columnspan=3, sticky="nsew", padx=5, pady=5)
        
        # 添加更多日志文件的按钮
        add_log_btn = ttk.Button(self.main_frame, text="Add Log File", command=self.add_log_file)
        add_log_btn.grid(row=2, column=0, columnspan=3, pady=5)
        
        # 结果输出区域
        output_frame = ttk.LabelFrame(self.main_frame, text="Output Settings", padding="10")
        output_frame.grid(row=3, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        
        ttk.Label(output_frame, text="Output Path:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.output_path_var = tk.StringVar()
        self.output_entry = ttk.Entry(output_frame, textvariable=self.output_path_var, width=50)
        self.output_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        gen_path_btn = ttk.Button(output_frame, text="Generate Path", command=self.generate_output_path)
        gen_path_btn.grid(row=0, column=2, padx=5, pady=5)
        
        # 状态和进度区域
        status_frame = ttk.Frame(self.main_frame)
        status_frame.grid(row=4, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(status_frame, textvariable=self.status_var)
        status_label.pack(side=tk.LEFT, padx=5)
        
        self.progress = ttk.Progressbar(status_frame, mode="indeterminate", length=300)
        self.progress.pack(side=tk.RIGHT, padx=5, fill=tk.X, expand=True)
        
        # 操作按钮区域
        button_frame = ttk.Frame(self.main_frame)
        button_frame.grid(row=5, column=0, columnspan=3, pady=10)
        
        compare_btn = ttk.Button(button_frame, text="Compare Logs", command=self.compare_logs)
        compare_btn.grid(row=0, column=0, padx=10)
        
        analyze_btn = ttk.Button(button_frame, text="Analyze Selected Log", command=self.analyze_selected_log)
        analyze_btn.grid(row=0, column=1, padx=10)
        
        # 结果文本区域
        result_frame = ttk.LabelFrame(self.main_frame, text="Results", padding="10")
        result_frame.grid(row=6, column=0, columnspan=3, sticky="nsew", padx=5, pady=5)
        self.main_frame.rowconfigure(6, weight=1)  # 让结果区域可以扩展
        
        # 设置更大的初始高度
        self.result_text = tk.Text(result_frame, wrap=tk.WORD, height=12)  # 增加高度
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(result_frame, command=self.result_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.config(yscrollcommand=scrollbar.set)
        
        # 创建可滚动区域用于日志文件输入
        self.canvas = tk.Canvas(self.logs_container)
        self.scrollbar = ttk.Scrollbar(self.logs_container, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # 修改：设置canvas的最小高度
        self.canvas.config(height=200)  # 为日志文件列表区域设置最小高度
        
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        # 添加：在日志容器和结果区域之间添加可调整大小的分隔线
        separator = ttk.Separator(self.main_frame, orient="horizontal")
        separator.grid(row=5, column=0, columnspan=3, sticky="ew", pady=2)
    def add_log_file(self):
        """添加一个新的日志文件输入行"""
        log_idx = len(self.log_files)
        
        # 创建新的日志文件框架
        log_frame = ttk.LabelFrame(self.scrollable_frame, text=f"Log File {log_idx+1}", padding="10")
        log_frame.pack(fill="x", padx=5, pady=5, expand=True)
        
        # 文件路径输入
        path_var = tk.StringVar()
        file_entry = ttk.Entry(log_frame, textvariable=path_var, width=50)
        file_entry.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        # 浏览按钮
        browse_btn = ttk.Button(
            log_frame, 
            text="Browse", 
            command=lambda idx=log_idx: self.browse_log_file(idx)
        )
        browse_btn.grid(row=0, column=1, padx=5, pady=5)
        
        # 删除按钮
        delete_btn = ttk.Button(
            log_frame, 
            text="Remove", 
            command=lambda idx=log_idx, frame=log_frame: self.remove_log_file(idx, frame)
        )
        delete_btn.grid(row=0, column=2, padx=5, pady=5)
        
        # 标签输入
        ttk.Label(log_frame, text="Label:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        label_var = tk.StringVar(value="log"+str(log_idx))
        label_entry = ttk.Entry(log_frame, textvariable=label_var, width=25)
        label_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        # 存储新添加的元素
        self.log_files.append("")
        self.log_entries.append((log_frame, path_var, file_entry))
        self.log_labels.append(label_var)
        
        # 更新UI
        self.root.update()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
        # 调整窗口大小以适应新添加的元素
        if log_idx > 1:  # 只在添加第三个及以后的日志时调整窗口大小
            current_height = self.root.winfo_height()
            if current_height < 700:  # 设置最大高度限制
                self.root.geometry(f"900x{min(current_height+60, 700)}")
    
    def remove_log_file(self, idx, frame):
        """移除一个日志文件输入行"""
        if len(self.log_files) <= 2:
            messagebox.showwarning("Warning", "At least two log files are required for comparison")
            return
        
        # 移除框架
        frame.destroy()
        
        # 从列表中移除
        del self.log_files[idx]
        del self.log_entries[idx]
        del self.log_labels[idx]
        
        # 重新编号剩余的日志框
        for i, (frame, _, _) in enumerate(self.log_entries):
            frame.config(text=f"Log File {i+1}")
        
        # 更新UI
        self.generate_output_path()
        self.root.update()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def browse_log_file(self, idx):
        """浏览选择日志文件"""
        filename = filedialog.askopenfilename(
            title=f"Select Log File {idx+1}", 
            filetypes=(("Log files", "*.log"), ("All files", "*.*"))
        )
        if filename:
            self.log_files[idx] = filename
            self.log_entries[idx][1].set(filename)
            self.generate_output_path()
    
    def generate_output_path(self):
        """根据选择的日志文件自动生成输出路径"""
        # 检查是否至少有两个日志文件
        valid_logs = [log for log in self.log_files if log]
        if len(valid_logs) < 2:
            return
        
        try:
            # 提取所有日志文件的父目录
            parent_dirs = []
            for log_file in valid_logs:
                parts = log_file.split(os.sep)
                if len(parts) >= 2:
                    parent_dirs.append(parts[-2])  # 获取日志文件的父目录名
            
            # 生成输出路径，包含所有日志文件的父目录名
            parent_dirs_str = "_".join(parent_dirs)
            if len(parent_dirs_str) > 30:  # 如果太长，进行截断
                parent_dirs_str = parent_dirs_str[:30] + "..."
                
            output_path = os.path.join("time_analysis_result", f"comparison_{parent_dirs_str}")
            self.output_path_var.set(output_path)
        except Exception as e:
            print(f"Error generating output path: {e}")
            self.output_path_var.set("time_analysis_result/comparison")
    
    def compare_logs(self):
        """比较选择的日志文件"""
        # 获取有效的日志文件路径
        valid_logs = []
        valid_labels = []
        for i, log_file in enumerate(self.log_files):
            if log_file:
                valid_logs.append(log_file)
                valid_labels.append(self.log_labels[i].get())
        
        if len(valid_logs) < 2:
            messagebox.showerror("Error", "Please select at least two log files to compare")
            return
        
        output_path = self.output_path_var.get()
        if not output_path:
            messagebox.showerror("Error", "Please specify an output path")
            return
        
        # 确保输出目录存在
        os.makedirs(output_path, exist_ok=True)
        
        # 更新状态
        self.status_var.set("Comparing logs...")
        self.progress.start()
        self.result_text.delete(1.0, tk.END)
        self.root.update()
        
        # 在新线程中执行比较
        thread = threading.Thread(target=self._run_multi_comparison, args=(valid_logs, valid_labels, output_path))
        thread.start()
    
    def _run_multi_comparison(self, log_files, labels, output_path):
        """在后台线程中执行多日志比较"""
        try:
            # 分析所有日志文件
            analyzers = []
            results = []
            hardware_times = []
            
            for log_file in log_files:
                analyzer = GPT2TimeAnalyzer(log_file)
                analyzers.append(analyzer)
                
                # 获取分析结果
                operation_times = analyzer.analyze_by_operation()
                hardware_time = analyzer.total_hardware_time
                
                results.append({
                    "operation_times": operation_times,
                    "total_hardware_time": hardware_time,
                    "total_generation_time": analyzer.total_generation_time
                })
                hardware_times.append(hardware_time)
            
            # 生成比较图表
            self._generate_multi_comparison_chart(log_files, labels, results, os.path.join(output_path, "operation_comparison.png"))
            
            # 计算性能改进
            base_time = hardware_times[0]
            improvement_text = ""
            for i in range(1, len(hardware_times)):
                improvement = (base_time - hardware_times[i]) / base_time * 100 if base_time > 0 else 0
                improvement_text += f"{labels[i]} vs {labels[0]}: {improvement:.2f}%\n"
            
            # 更新结果文本
            self.root.after(0, lambda: self._update_result_text(
                f"===== Multiple Log Comparison Results =====\n\n" +
                "\n".join([f"{labels[i]}: {os.path.basename(log_files[i])}\n"
                         f"Total Hardware Time: {hardware_times[i]:.4f} seconds"
                         for i in range(len(log_files))]) +
                f"\n\nPerformance Improvements (relative to {labels[0]}):\n{improvement_text}\n" +
                f"Comparison chart saved to:\n{os.path.join(output_path, 'operation_comparison.png')}\n\n" +
                f"Full results saved in:\n{output_path}"
            ))
            
            # 生成详细报告
            self._save_multi_comparison_report(log_files, labels, results, os.path.join(output_path, "comparison_report.txt"))
            
        except Exception as e:
            error_msg = f"Error during comparison: {str(e)}\n\n{traceback.format_exc()}"
            self.root.after(0, lambda: self._update_result_text(error_msg))
        finally:
            # 恢复UI状态
            self.root.after(0, self._reset_ui)
    
    def _generate_multi_comparison_chart(self, log_files, labels, results, save_path):
        """生成多日志比较图表"""
        # 获取所有操作类型
        all_ops = set()
        for result in results:
            all_ops.update(result["operation_times"].keys())
        
        # 排序操作类型并添加总时间
        operation_order = ['QKV计算', '注意力投影', 'MLP第一层', 'MLP第二层', '反量化', '总时间'] #可加 权重加载
        ops = []
        
        # 临时创建一个分析器以获取映射
        analyzer = GPT2TimeAnalyzer(log_files[0])
        
        for op in operation_order:
            if op == '总时间' or op in all_ops:
                if op == '总时间':
                    ops.append("Total Time")
                else:
                    ops.append(analyzer.operation_map.get(op, op))
                
        # 准备数据
        op_data = []
        for op in operation_order:
            if op == '总时间':
                # 为"总时间"项获取数据
                total_values = []
                for result in results:
                    total_values.append(result["total_hardware_time"])
                op_data.append(total_values)
            elif op in all_ops:
                op_values = []
                for result in results:
                    op_values.append(result["operation_times"].get(op, 0))
                op_data.append(op_values)
        
        # 创建图表
        fig, ax = plt.subplots()  # 增加图表尺寸以容纳更多条形
        x = np.arange(len(ops))
        
        # 为每组柱状图设置宽度和间隔参数 - 增加间距
        group_width = 1.4  # 每组柱状图的总宽度
        bar_width = group_width / len(results)/1.5 # 单个柱子的宽度
        group_spacing = 0.3  # 增加组间距离
        
        # 可视化颜色
        colors = ['#FF5252', '#42A5F5', '#66BB6A', '#FFC107', '#9C27B0', '#FF9800', '#607D8B']
        
        # 绘制条形图
        for i in range(len(results)):
            values = [data[i] for data in op_data]
            # 计算每个柱子的位置，使同组柱子聚集在一起，并增加间距
            offset = bar_width * i - group_width / 2 + bar_width / 2
            bars = ax.bar(x + offset, values, bar_width * 0.85, label=labels[i], color=colors[i % len(colors)])
            
            # 在柱上标注具体的时间值（保留两位小数）
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',  # 修改为保留两位小数
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3点垂直偏移
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=6,
                            fontweight='bold')
        
        # 添加图表元素
        ax.set_ylabel('Time (ms)', fontweight='bold', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(ops, rotation=45, ha='right', fontsize=10)
        ax.legend()
        
        # 设置较大值的y轴限制
        ax.set_ylim(0, max([max(data) for data in op_data]) * 1.15)
        
        fig.tight_layout()
        
        # 保存图表
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        # 保存绘图数据到CSV文件
        data_dict = {}
        # 为每个标签创建一列数据
        for i, label in enumerate(labels):
            data_dict[label] = [data[i] for data in op_data]
        
        # 创建DataFrame
        df = pd.DataFrame(data_dict, index=ops)
        
        # 确定CSV保存路径（与图像相同目录）
        csv_path = os.path.splitext(save_path)[0] + "_data.csv"
        
        # 保存为CSV
        df.to_csv(csv_path)
        print(f"Chart data saved to CSV: {csv_path}")        
    def _save_multi_comparison_report(self, log_files, labels, results, filepath):
        """保存多日志比较结果到文本报告"""
        with open(filepath, 'w') as f:
            f.write("===== GPT-2 Model Time Analysis Multi-Comparison Report =====\n\n")
            
            # 文件信息
            for i, log_file in enumerate(log_files):
                f.write(f"{labels[i]}: {log_file}\n")
            f.write("\n")
            
            # 时间摘要
            hardware_times = [result["total_hardware_time"] for result in results]
            base_time = hardware_times[0]
            
            f.write("===== Performance Summary =====\n")
            for i, time in enumerate(hardware_times):
                f.write(f"{labels[i]} Total Hardware Time: {time:.4f} millisecond\n")
            
            f.write("\n===== Performance Improvements =====\n")
            for i in range(1, len(hardware_times)):
                improvement = base_time / hardware_times[i] * 100 if base_time > 0 else 0
                f.write(f"{labels[i]} vs {labels[0]}: {improvement:.2f}%\n")
            f.write("\n")
            
            # 操作时间比较
            f.write("===== Operation Time Comparison =====\n")
            header = f"{'Operation':<20}"
            for label in labels:
                header += f" {label+' (ms)':<15}"
            f.write(header + "\n")
            f.write("-" * (20 + 15 * len(labels)) + "\n")
            
            # 获取所有操作类型
            all_ops = set()
            for result in results:
                all_ops.update(result["operation_times"].keys())
            
            # 使用一个一致的操作顺序
            operation_order = ['QKV计算', '注意力投影', 'MLP第一层', 'MLP第二层', '反量化','总时间'] #可加权重加载
            analyzer = GPT2TimeAnalyzer(log_files[0])  # 临时创建一个分析器以获取映射
            
            for op in operation_order:
                if op in all_ops:
                    eng_op = analyzer.operation_map.get(op, op)
                    line = f"{eng_op:<20}"
                    for result in results:
                        time = result["operation_times"].get(op, 0)
                        line += f" {time:<15.2f}"
                    f.write(line + "\n")
            
            f.write("\n===== End of Report =====\n")
        
        print(f"Detailed multi-comparison report saved to {filepath}")
    
    def analyze_selected_log(self):
        """分析选中的日志文件"""
        # 弹出对话框，让用户选择要分析的日志文件
        options = []
        for i, log_file in enumerate(self.log_files):
            if log_file:
                options.append(f"{i+1}: {os.path.basename(log_file)}")
        
        if not options:
            messagebox.showerror("Error", "No log files available for analysis")
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Log to Analyze")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Select a log file to analyze:", font=("Arial", 12)).pack(pady=10)
        
        var = tk.StringVar(value=options[0] if options else "")
        listbox = tk.Listbox(dialog, listvariable=tk.StringVar(value=options), height=10, width=50)
        listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        def on_select():
            selection = listbox.curselection()
            if selection:
                idx = int(options[selection[0]].split(":")[0]) - 1
                output_dir = os.path.join(self.output_path_var.get(), f"{self.log_labels[idx].get()}_analysis")
                dialog.destroy()
                self._analyze_log(self.log_files[idx], output_dir)
        
        ttk.Button(dialog, text="Analyze", command=on_select).pack(pady=10)
    
    def _analyze_log(self, log_file, output_dir):
        """分析单个日志文件"""
        if not log_file:
            messagebox.showerror("Error", "Please select a valid log file")
            return
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 更新状态
        self.status_var.set(f"Analyzing {os.path.basename(log_file)}...")
        self.progress.start()
        self.result_text.delete(1.0, tk.END)
        self.root.update()
        
        # 在新线程中执行分析
        thread = threading.Thread(target=self._run_analysis, args=(log_file, output_dir))
        thread.start()
    
    def _run_analysis(self, log_file, output_dir):
        """在后台线程中执行日志分析"""
        try:
            # 创建分析器实例
            analyzer = GPT2TimeAnalyzer(log_file)
            
            # 执行分析
            detailed_layers = [0, 5, 11]  # 详细分析特定层
            results = analyzer.analyze_all(output_dir=output_dir, detail_layers=detailed_layers)
            
            # 更新结果文本
            self.root.after(0, lambda: self._update_result_text(
                f"===== Analysis Results for {os.path.basename(log_file)} =====\n\n" +
                f"Total Hardware Computation Time: {results['total_hardware_time']:.4f} millisecond\n\n" +
                f"===== Operation Times =====\n" +
                f"Total Text Generation Time: {results['total_generation_time']:.4f} seconds\n" +
                f"Hardware Acceleration Ratio: {(results['total_hardware_time']/results['total_generation_time']*0.1):.2f}%\n\n" +
                f"Most time-consuming operations:\n" + 
                "\n".join([f"- {analyzer.operation_map.get(op, op)}: {time:.2f} ms" 
                          for op, time in sorted(results['operations'].items(), key=lambda x: x[1], reverse=True)[:3]]) +
                f"\n\nFull analysis results saved in:\n{output_dir}"
            ))
            
        except Exception as e:
            error_msg = f"Error during analysis: {str(e)}\n\n{traceback.format_exc()}"
            self.root.after(0, lambda: self._update_result_text(error_msg))
        finally:
            # 恢复UI状态
            self.root.after(0, self._reset_ui)
    
    def _update_result_text(self, text):
        """更新结果文本框"""
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, text)
    
    def _reset_ui(self):
        """重置UI状态"""
        self.progress.stop()
        self.status_var.set("Ready")



if __name__ == "__main__":
    # GUI模式
    root = tk.Tk()
    app = TimeAnalyzerGUI(root)
    root.mainloop()
