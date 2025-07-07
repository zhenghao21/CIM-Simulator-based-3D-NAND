# model_monitor.py
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import torch
import torch.nn.functional as F
import pandas as pd  # 添加pandas库用于Excel文件操作

class ModelMonitor:
    """
    模型运行过程监测工具，记录计算过程中的中间结果
    """
    
    def __init__(self, output_dir):
        """初始化监测工具"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 存储计算结果的字典
        self.computation_records = defaultdict(dict)
        self.current_step = 0
        
        # 定义全局字体样式
        self.font_properties = {
            'family': 'Arial',
            'weight': 'bold',
            'size': 10
        }
        
        # 设置matplotlib全局字体
        plt.rc('font', **self.font_properties)
        plt.rc('axes', labelweight='bold')
        
        print(f"模型监测工具已初始化，输出目录: {output_dir}")
    
    def set_step(self, step):
        """设置当前处理的token步骤"""
        self.current_step = step
        print(f"监测工具: 设置当前步骤为 {step}")
    
    def record(self, key, value):
        """记录一个计算结果"""
        # 确保value是numpy数组
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        
        self.computation_records[self.current_step][key] = value
        # print(f"记录数据: step={self.current_step}, key={key}, shape={value.shape}")
    
    def _setup_plot_style(self, ax=None):
        """设置图表的通用样式"""
        if ax is None:
            ax = plt.gca()
        
        # 启用次刻度线
        ax.minorticks_on()
        
        # 设置主刻度线和次刻度线方向为向内
        ax.tick_params(which='both', direction='in')
        
        # 次刻度线显示在所有边上
        ax.tick_params(which='minor', top=True, right=True)
        ax.tick_params(which='major', top=True, right=True)
        
    def plot_token_probabilities(self, probs, token_text, token_id, tokenizer=None):
        """
        绘制token概率分布
        
        Args:
            probs: 概率分布
            token_text: 当前token的文本
            token_id: 当前token的ID
            tokenizer: 用于将token ID转换为文本的tokenizer对象
        """
        if isinstance(probs, torch.Tensor):
            probs = probs.detach().cpu().numpy()
            
        # 获取前20个最高概率的token对应的索引
        top_indices = np.argsort(probs)[-20:][::-1]
        top_probs = probs[top_indices]
        
        # 准备标签文本
        if tokenizer is not None:
            # 如果有tokenizer，将索引转换为文本
            token_labels = []
            for idx in top_indices:
                try:
                    token_str = tokenizer.decode([idx]).strip()
                    # 处理特殊或不可见字符，使其更好显示
                    if not token_str or token_str.isspace():
                        token_str = f"[ID:{idx}]"
                    elif len(token_str) > 10:  # 如果文本太长，截断
                        token_str = token_str[:10] + "..."
                    token_labels.append(token_str)
                except:
                    token_labels.append(f"[ID:{idx}]")
        else:
            # 如果没有tokenizer，仍然显示ID
            token_labels = [f"{i}" for i in top_indices]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(range(len(top_indices)), top_probs, color='orange')
        ax.set_xticks(range(len(top_indices)))
        ax.set_xticklabels(token_labels, rotation=45, ha='right')
        ax.set_xlabel('Token')
        ax.set_ylabel('Probability')
        # ax.set_title(f'Step {self.current_step}: Token "{token_text}" (ID: {token_id}) Probability Distribution')
        
        # 应用通用样式
        self._setup_plot_style(ax)
        
        plt.tight_layout()
        
        # 保存图表
        save_path = os.path.join(self.output_dir, f"prob_step_{self.current_step}_token_{token_id}.png")
        plt.savefig(save_path, dpi=600)
        plt.close()
        
        # 保存原始概率分布
        np.save(os.path.join(self.output_dir, f"prob_step_{self.current_step}_token_{token_id}.npy"), probs)
        
        # 保存数据到Excel文件
        self.save_token_probs_to_excel(top_indices, top_probs, token_labels, token_text, token_id)
    
    def save_token_probs_to_excel(self, token_indices, token_probs, token_labels, token_text, token_id):
        """
        将token概率数据保存到Excel文件
        
        Args:
            token_indices: token的索引数组
            token_probs: token的概率数组
            token_labels: token的文本标签数组
            token_text: 当前token的文本
            token_id: 当前token的ID
        """
        # 创建数据框
        data = {
            'Token ID': token_indices,
            'Token Text': token_labels,
            'Probability': token_probs
        }
        df = pd.DataFrame(data)
        
        # 添加元数据
        metadata = pd.DataFrame({
            'Info': ['Current Step', 'Token Text', 'Token ID'],
            'Value': [self.current_step, token_text, token_id]
        })
        
        # 保存到Excel文件
        excel_path = os.path.join(self.output_dir, f"prob_step_{self.current_step}_token_{token_id}.xlsx")
        
        with pd.ExcelWriter(excel_path) as writer:
            df.to_excel(writer, sheet_name='Probability Data', index=False)
            metadata.to_excel(writer, sheet_name='Metadata', index=False)
        
        print(f"Token概率数据已保存到Excel文件: {excel_path}")
    
    def calculate_metrics(self, original, quantized):
        """计算原始值和量化值之间的差异指标"""
        if isinstance(original, torch.Tensor):
            original = original.detach().cpu().numpy()
        if isinstance(quantized, torch.Tensor):
            quantized = quantized.detach().cpu().numpy()
            
        diff = original - quantized
        abs_diff = np.abs(diff)
        
        metrics = {
            'mae': np.mean(abs_diff),                     # 平均绝对误差
            'max_abs_error': np.max(abs_diff),            # 最大绝对误差
            'mse': np.mean(np.square(diff)),              # 均方误差
            'rmse': np.sqrt(np.mean(np.square(diff))),    # 均方根误差
            'relative_error': np.mean(abs_diff / (np.abs(original) + 1e-10)),  # 相对误差
            'cosine_similarity': np.sum(original * quantized) / (np.sqrt(np.sum(original**2)) * np.sqrt(np.sum(quantized**2)) + 1e-10)  # 余弦相似度
        }
        
        return metrics
    
    def compare_original_vs_quantized(self):
        """比较原始浮点计算和量化计算的结果"""
        metrics_summary = {}
        
        for step in self.computation_records:
            step_metrics = {}
            
            # 分析各层各组件的计算差异
            for layer_idx in range(12):  # 假设最多12层
                layer_name = f"layer_{layer_idx}"
                
                # 比较Q、K、V等计算结果
                for component in ['q', 'k', 'v']:
                    raw_key = f"{layer_name}_{component}_software"
                    quan_key = f"{layer_name}_{component}_hardware"
                    if raw_key in self.computation_records[step] and quan_key in self.computation_records[step]:
                        raw_data = self.computation_records[step][raw_key]
                        quan_data = self.computation_records[step][quan_key]
                        step_metrics[f"{layer_name}_{component}"] = self.calculate_metrics(raw_data, quan_data)
                
                # 比较注意力输出投影结果
                raw_key = f"{layer_name}_attn_output_software"
                quan_key = f"{layer_name}_attn_output_hardware"
                if raw_key in self.computation_records[step] and quan_key in self.computation_records[step]:
                    raw_data = self.computation_records[step][raw_key]
                    quan_data = self.computation_records[step][quan_key]
                    step_metrics[f"{layer_name}_attn_proj"] = self.calculate_metrics(raw_data, quan_data)
                
                # 比较MLP的FC1输出结果
                raw_key = f"{layer_name}_mlp_fc1_output_software"
                quan_key = f"{layer_name}_mlp_fc1_output_hardware"
                if raw_key in self.computation_records[step] and quan_key in self.computation_records[step]:
                    raw_data = self.computation_records[step][raw_key]
                    quan_data = self.computation_records[step][quan_key]
                    step_metrics[f"{layer_name}_mlp_fc1"] = self.calculate_metrics(raw_data, quan_data)
                
                # 比较激活后的结果
                raw_key = f"{layer_name}_mlp_activated_software"
                quan_key = f"{layer_name}_mlp_activated_hardware"
                if raw_key in self.computation_records[step] and quan_key in self.computation_records[step]:
                    raw_data = self.computation_records[step][raw_key]
                    quan_data = self.computation_records[step][quan_key]
                    step_metrics[f"{layer_name}_mlp_activated"] = self.calculate_metrics(raw_data, quan_data)
                
                # 比较MLP的FC2输出结果
                raw_key = f"{layer_name}_mlp_output_software"
                quan_key = f"{layer_name}_mlp_output_hardware"
                if raw_key in self.computation_records[step] and quan_key in self.computation_records[step]:
                    raw_data = self.computation_records[step][raw_key]
                    quan_data = self.computation_records[step][quan_key]
                    step_metrics[f"{layer_name}_mlp_fc2"] = self.calculate_metrics(raw_data, quan_data)
            
            metrics_summary[step] = step_metrics
        
        # 保存指标摘要
        with open(os.path.join(self.output_dir, "metrics_summary.txt"), "w") as f:
            for step, step_metrics in metrics_summary.items():
                f.write(f"步骤 {step}:\n")
                for component, metrics in step_metrics.items():
                    f.write(f"  {component}:\n")
                    for metric_name, value in metrics.items():
                        f.write(f"    {metric_name}: {value:.6f}\n")
                f.write("\n")
        
        # 创建可视化图表
        for step, step_metrics in metrics_summary.items():
            if not step_metrics:  # 如果没有数据，跳过
                continue
                
            fig, ax = plt.subplots(figsize=(15, 8))
            
            # 提取所有组件的MAE
            components = []
            mae_values = []
            
            for component, metrics in step_metrics.items():
                components.append(component)
                mae_values.append(metrics['mae'])
            
            # 绘制柱状图
            ax.bar(components, mae_values, color='orange')
            # plt.title(f"Mean Absolute Error of Algorithm Modules (step {step})")
            ax.set_xlabel("Algorithm Modules")
            ax.set_ylabel("Mean Absolute Error (MAE)")
            ax.set_xticklabels(components, rotation=90)
            
            # 应用通用样式
            self._setup_plot_style(ax)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"mae_step_{step}.png"), dpi=600)
            plt.close()
            
            # 绘制误差累计效应图
            self._plot_error_accumulation(step_metrics, step)
            
            # 保存误差数据到Excel
            self.save_metrics_to_excel(step_metrics, step)
        
        print(f"量化分析完成，结果已保存至 {self.output_dir}")
        return metrics_summary
    
    def save_metrics_to_excel(self, step_metrics, step):
        """将误差指标数据保存到Excel文件中"""
        # 准备所有指标数据
        components = []
        metrics_data = {
            'mae': [], 
            'max_abs_error': [], 
            'mse': [], 
            'rmse': [], 
            'relative_error': [], 
            'cosine_similarity': []
        }
        
        for component, metrics in step_metrics.items():
            components.append(component)
            for metric_name in metrics_data.keys():
                metrics_data[metric_name].append(metrics[metric_name])
        
        # 创建数据框
        df = pd.DataFrame({'Component': components})
        for metric_name, values in metrics_data.items():
            df[metric_name] = values
        
        # 保存到Excel文件
        excel_path = os.path.join(self.output_dir, f"metrics_step_{step}.xlsx")
        df.to_excel(excel_path, index=False)
        print(f"指标数据已保存到Excel文件: {excel_path}")
    
    def _plot_error_accumulation(self, step_metrics, step):
        """绘制误差在层间的累计效应"""
        # 按层索引聚合误差
        layer_mae = {}
        component_types = {}
        
        for component, metrics in step_metrics.items():
            parts = component.split('_')
            if len(parts) >= 2 and parts[0] == 'layer':
                layer_idx = int(parts[1])
                comp_type = '_'.join(parts[2:])
                
                if layer_idx not in layer_mae:
                    layer_mae[layer_idx] = {}
                    
                layer_mae[layer_idx][comp_type] = metrics['mae']
                component_types[comp_type] = True
        
        if not layer_mae:
            return
            
        # 绘制每种组件类型的误差趋势
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 定义颜色循环以确保一致的颜色
        colors = plt.cm.tab10(np.linspace(0, 1, len(component_types)))
        
        layers = sorted(layer_mae.keys())
        for i, comp_type in enumerate(sorted(component_types.keys())):
            values = [layer_mae[layer].get(comp_type, 0) for layer in layers]
            ax.plot(layers, values, 'o-', label=comp_type, color=colors[i])
            
        # plt.title(f"The errors of layers (step{step})")
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Mean Absolute Error (MAE)")
        ax.legend()
        
        # 应用通用样式
        self._setup_plot_style(ax)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"error_accumulation_step_{step}.png"), dpi=600)
        plt.close()
        
        # 保存层间误差数据到Excel
        self.save_layer_errors_to_excel(layer_mae, component_types, layers, step)
    
    def save_layer_errors_to_excel(self, layer_mae, component_types, layers, step):
        """将层间误差数据保存到Excel文件"""
        # 创建数据框
        data = {'Layer': layers}
        
        # 为每种组件类型添加一列
        for comp_type in sorted(component_types.keys()):
            values = [layer_mae[layer].get(comp_type, 0) for layer in layers]
            data[comp_type] = values
        
        df = pd.DataFrame(data)
        
        # 保存到Excel文件
        excel_path = os.path.join(self.output_dir, f"layer_errors_step_{step}.xlsx")
        df.to_excel(excel_path, index=False)
        print(f"层间误差数据已保存到Excel文件: {excel_path}")
    
    def save_computation_records(self):
        """保存计算记录到磁盘"""
        records_dir = os.path.join(self.output_dir, "records")
        os.makedirs(records_dir, exist_ok=True)
        
        for step, step_records in self.computation_records.items():
            step_dir = os.path.join(records_dir, f"step_{step}")
            os.makedirs(step_dir, exist_ok=True)
            
            # 保存该步骤的所有记录
            for name, value in step_records.items():
                # 跳过太大的数据以节省空间
                if any(x in name for x in ['multi_head']):
                    continue
                    
                np.save(os.path.join(step_dir, f"{name}.npy"), value)
        
        print(f"计算记录已保存到 {records_dir}")