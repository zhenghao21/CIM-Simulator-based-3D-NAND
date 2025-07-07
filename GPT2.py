import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
import numpy as np
from hardware_block_epoch_last import NANDHardwareSimulator
import matplotlib
import math
import time
from PyQt6.QtWidgets import QApplication
from datetime import datetime
from model_monitor import ModelMonitor
from hardware_visualizer import HardwareVisualizer
import config

# 配置matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
matplotlib.rcParams['axes.unicode_minus'] = False 


class HardwareAcceleratedGPT2:
    """
    使用NANDHardwareSimulator加速的GPT-2模型
    """
    
    def __init__(self,logger, log_dir):
        """
        初始化硬件加速的GPT-2
        
        参数:
        - load_from_cache: 是否从缓存加载权重
        - enable_monitoring: 是否启用监控
        """
        # 使用配置参数或传入参数
        self.enable_monitoring = config.ENABLE_MONITORING
        self.logger=logger
        # 初始化监控器
        if self.enable_monitoring:
            self.monitor = ModelMonitor(
            log_dir
            )
        else:
            self.monitor = None
            

        # 加载GPT-2 tokenizer和模型
        self.logger.info("加载GPT-2 tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        self.logger.info("加载GPT-2模型和权重...")
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")

        # 存储一些GPT-2的配置参数
        self.config = self.model.config
        
        # 初始化硬件模拟器
        self.hardware_sim = NANDHardwareSimulator(
            self.config.n_embd,
            self.logger
        )
        # 添加时间统计变量
        self.total_time = config.WL_SETUP_TIME_BASE*self.hardware_sim.blocks_per_operation
        self.total_power=0
        self.total_sensing_power=0
        # 确保n_inner存在，如果不存在，使用默认值(通常是隐藏层大小的4倍)
        if not hasattr(self.config, 'n_inner') or self.config.n_inner is None:
            self.logger.warning("配置中找不到n_inner参数，使用n_embd的4倍作为默认值")
            self.config.n_inner = 4 * self.config.n_embd

        self.logger.info("将权重转移到硬件模拟器...")
        self._load_weights_to_hardware()          

    def _hardware_processing(self,model_name,hidden_input,info_name):
        """
        在硬件上执行GPT-2的前向传播
        """
        if isinstance(hidden_input, torch.Tensor):           
            hidden_input=hidden_input.numpy()
        if config.SYMMETRIC:
            self.logger.info("开始硬件处理...")
            input_result, input_scales,zero_point_input= self.hardware_sim.set_input(hidden_input,symmetric=config.SYMMETRIC)
            model_outputs, model_time,model_power,model_sensing_power = self.hardware_sim.compute(
                input_result, 
                matrix_name=model_name,
                total_compute_time=self.total_time
            )
            self.before_time=self.total_time
            self.total_time = model_time
            self.total_power += model_power
            self.total_sensing_power+=model_sensing_power
            
            self.logger.info(f'计算{info_name}，耗时{self.total_time-self.before_time}us，目前总耗时{self.total_time}us')
            self.logger.info(f'计算{info_name}，耗能{model_power}uJ，目前总耗能{self.total_power}uJ')
            self.logger.info(f'计算{info_name}，{info_name}耗能{model_power}中有{model_sensing_power}uJ用于sensing，目前总sensing耗能{self.total_sensing_power}uJ')
            model_scale = self.weight_locations[model_name]['scale']
            model_Weight_scales = np.matmul(input_scales.reshape(-1, 1), model_scale.reshape(1, -1))
            output = torch.tensor((model_outputs[model_name] * model_Weight_scales),dtype=torch.float32)
        else:
            # 非对称量化
            self.logger.info("开始硬件处理...")
            input_result, input_scales,zero_point_input= self.hardware_sim.set_input(hidden_input,symmetric=config.SYMMETRIC)
            model_outputs, model_time,model_power,model_sensing_power = self.hardware_sim.compute(
                input_result, 
                matrix_name=model_name,
                total_compute_time=self.total_time
            )
            self.before_time=self.total_time
            self.total_time = model_time
            self.total_power += model_power
            self.total_sensing_power+=model_sensing_power
            self.logger.info(f'计算{info_name}，耗时{self.total_time-self.before_time}us，目前总耗时{self.total_time}us')
            self.logger.info(f'计算{info_name}，耗能{model_power}uJ，目前总耗能{self.total_power}uJ')
            self.logger.info(f'计算{info_name}，{info_name}耗能{model_power}中有{model_sensing_power}uJ用于sensing，目前总sensing耗能{self.total_sensing_power}uJ')

            one_matrix = np.ones((1,hidden_input.shape[-1]))
            one_matrix_result, _,zero_point=self.hardware_sim.set_input(one_matrix)
            Dequantize_out,dequant_times,dequant_power,dequant_sensing_power= self.hardware_sim.compute(
                one_matrix_result, 
                matrix_name=model_name,
                total_compute_time=self.total_time
            )
            self.before_time = self.total_time
            self.total_time = dequant_times
            self.total_power+= dequant_power
            self.total_sensing_power+= dequant_sensing_power
            
            self.logger.info(f'计算{info_name}反量化，耗时{self.total_time-self.before_time}us，目前总耗时{self.total_time}us')
            self.logger.info(f'计算{info_name}反量化，耗能{dequant_power}uJ，目前总耗能{self.total_power}uJ') 
            self.logger.info(f'计算{info_name}反量化，{info_name}反量化耗能{dequant_power}中有{dequant_sensing_power}uJ用于sensing，目前总sensing耗能{self.total_sensing_power}uJ')

            # 处理输出
            model_scale = self.weight_locations[model_name]['scale']
            Dequantize_out=Dequantize_out[model_name]
            col_sum = Dequantize_out.reshape(-1)//(2**(config.QUANTIZATION_BITS-1)-1)
            scale_matrix = np.outer(input_scales, model_scale) 
            correction = np.outer(zero_point_input, col_sum)  # 形状为[m, n]
            output = torch.tensor((scale_matrix * (model_outputs[model_name] - correction)),dtype=torch.float32)
        return output
    def _load_weights_to_hardware(self):
        """
        将GPT-2的权重加载到硬件模拟器中
        """
        self.logger.info("开始将权重加载到硬件模拟器...")
        
        # 准备起始位置
        start_location = {
            'layer': 0,
            'plane': 0,
            'block': 0,
            'tsg': 0,
            'bl': 0,
            'block_epochs': 0
        }
        
        # 存储权重位置信息的字典
        self.weight_locations = {}
        
        # 遍历GPT-2的每一层
        for layer_idx, layer in enumerate(self.model.transformer.h):
            # 处理自注意力权重
            # Query weights
            qkv_weights = layer.attn.c_attn.weight.detach().numpy()
            self.monitor.record(f"layer_{layer_idx}_qkv_weight", qkv_weights)
            qkv_name = f"L{layer_idx}_Wqkv"
            qkv_scale, end_loc,qkv_weights_Q = self.hardware_sim.load_weights(
                qkv_weights,
                location=start_location,
                matrix_name=qkv_name
            )
            self.weight_locations[qkv_name] = {'scale': qkv_scale, 'shape': qkv_weights.shape[-1],'Quant':qkv_weights_Q}
            start_location = end_loc
            self.logger.debug(f"剩余存储空间占比：{self.hardware_sim.rest_space/self.hardware_sim.total_space}")
            # Projection weights
            proj_weights = layer.attn.c_proj.weight.detach().numpy()
            self.monitor.record(f"layer_{layer_idx}_proj_weights", proj_weights)
            proj_name = f"L{layer_idx}_Wproj"
            proj_scale, end_loc,proj_weights_Q = self.hardware_sim.load_weights(
                proj_weights,
                location=start_location,
                matrix_name=proj_name
            )
            self.weight_locations[proj_name] = {'scale': proj_scale, 'shape': proj_weights.shape[-1],'Quant':proj_weights_Q}
            start_location = end_loc
            self.logger.debug(f"剩余存储空间占比：{self.hardware_sim.rest_space/self.hardware_sim.total_space}")
            # 处理前馈网络权重
            # First dense layer
            fc1_weights = layer.mlp.c_fc.weight.detach().numpy()
            self.monitor.record(f"layer_{layer_idx}_fc1_weights", fc1_weights)
            fc1_name = f"L{layer_idx}_Wfc1"
            fc1_scale, end_loc, fc1_weights_Q = self.hardware_sim.load_weights(
                fc1_weights,
                location=start_location,
                matrix_name=fc1_name
            )
            self.weight_locations[fc1_name] = {'scale': fc1_scale, 'shape': fc1_weights.shape[-1],'Quant':fc1_weights_Q}
            start_location = end_loc
            self.logger.debug(f"剩余存储空间占比：{self.hardware_sim.rest_space/self.hardware_sim.total_space}")
            # Second dense layer
            fc2_weights = layer.mlp.c_proj.weight.detach().numpy()
            self.monitor.record(f"layer_{layer_idx}_fc2_weights", fc2_weights)
            fc2_name = f"L{layer_idx}_Wfc2"
            fc2_scale, end_loc,fc2_weights_Q = self.hardware_sim.load_weights(
                fc2_weights,
                location=start_location,
                matrix_name=fc2_name
            )
            self.weight_locations[fc2_name] = {'scale': fc2_scale, 'shape': fc2_weights.shape[-1],'Quant':fc2_weights_Q}
            start_location = end_loc
            self.logger.debug(f"剩余存储空间占比：{self.hardware_sim.rest_space/self.hardware_sim.total_space}")
        self.logger.info("权重加载完成。")
        self.total_time = self.total_time + self.hardware_sim.write_time
        self.logger.info(f'加载权重，耗时{self.hardware_sim.write_time}，目前总耗时{self.total_time}')
        
    def forward_attention(self, hidden_states:torch.tensor, layer_idx, past_kv=None, use_cache=False):
        """
        使用硬件模拟器执行注意力计算，支持KV缓存和完全硬件加速
        
        参数:
        - hidden_states: 输入隐藏状态
        - layer_idx: 层索引
        - past_kv: 过去层的key和value缓存
        - use_cache: 是否使用和返回缓存
        
        返回:
        - attention_output: 注意力层的输出
        - present_kv: 当前层的key和value缓存(如果use_cache=True)
        """
        # 监测输入
        if self.enable_monitoring and self.monitor:
            self.monitor.record(f"layer_{layer_idx}_input", hidden_states)
            
        # 获取当前层的GPT-2实现，用于访问某些参数和偏置值
        layer = self.model.transformer.h[layer_idx]
        # 初始化present_kv，用于存储当前的key和value缓存
        present_kv = None
        
        # # 确定是只处理新token还是处理整个序列
        if use_cache and past_kv is not None:
            # 如果使用缓存，我们只需处理新的token
            past_k, past_v = past_kv
            batch_size, seq_len, hidden_size = hidden_states[:, -1:, :].shape
            # 为硬件模拟器准备输入 - 只处理新的token（最后一个token）
            hidden_flat = hidden_states[:, -1:, :].reshape(-1, hidden_size)
            
        else:
            # 不使用缓存，处理整个序列
            batch_size, seq_len, hidden_size = hidden_states.shape
            hidden_flat = hidden_states.reshape(-1, hidden_size)
        
        # 添加软件计算QKV部分
        if self.enable_monitoring and self.monitor:
            # 获取QKV权重和偏置
            qkv_weight = layer.attn.c_attn.weight
            qkv_bias = layer.attn.c_attn.bias if hasattr(layer.attn.c_attn, 'bias') and layer.attn.c_attn.bias is not None else None
            
            # 软件计算QKV
            qkv_software = torch.matmul(hidden_flat, qkv_weight)
            if qkv_bias is not None:
                qkv_software = qkv_software + qkv_bias
                
            # 分离QKV
            q_software, k_software, v_software = qkv_software.chunk(3, dim=1)
            
            # 记录软件计算结果
            self.monitor.record(f"layer_{layer_idx}_q_software", q_software.reshape(batch_size, seq_len, -1))
            self.monitor.record(f"layer_{layer_idx}_k_software", k_software.reshape(batch_size, seq_len, -1))
            self.monitor.record(f"layer_{layer_idx}_v_software", v_software.reshape(batch_size, seq_len, -1))
            
        # 计算QKV (硬件)
        qkv=self._hardware_processing(f'L{layer_idx}_Wqkv',hidden_flat,'QKV')
        
        # 添加QKV偏置
        if hasattr(layer.attn.c_attn, 'bias') and layer.attn.c_attn.bias is not None:
            qkv += layer.attn.c_attn.bias 
            self.monitor.record(f"layer_{layer_idx}_qkv_bias", layer.attn.c_attn.bias)
        
        # 分离QKV
        q, k, v = qkv.chunk(3, dim=1)
        q=q.reshape(batch_size, seq_len, self.config.n_embd)
        k=k.reshape(batch_size, seq_len, self.config.n_embd)
        v=v.reshape(batch_size, seq_len, self.config.n_embd)
        
        # 监测QKV结果
        if self.enable_monitoring and self.monitor:
            self.monitor.record(f"layer_{layer_idx}_q_hardware", q)
            self.monitor.record(f"layer_{layer_idx}_k_hardware", k)
            self.monitor.record(f"layer_{layer_idx}_v_hardware", v)
            
        # 如果使用缓存，则拼接当前的key和value和过去的缓存
        if use_cache and past_kv is not None:
            past_k, past_v = past_kv
            # 拼接缓存
            k = torch.cat((past_k, k), 1)
            v = torch.cat((past_v, v), 1)
            # 创建present_kv
            present_kv = (k, v)
        elif use_cache:
            # 没有历史缓存但要使用缓存，直接将当前的k和v作为缓存
            present_kv = (k, v)
        
        # 实现多头注意力
        head_dim = self.config.n_embd // self.config.n_head
        
        # 1. 重塑Q、K、V为多头形式 [batch_size, seq_len, n_head, head_dim]
        q_multi_head = q.reshape(batch_size, seq_len, self.config.n_head, head_dim)
        k_multi_head = k.reshape(batch_size, k.shape[1], self.config.n_head, head_dim)  
        v_multi_head = v.reshape(batch_size, v.shape[1], self.config.n_head, head_dim)
        
        # 2. 转置为 [batch_size, n_head, seq_len, head_dim]
        q_multi_head = q_multi_head.transpose(1,2)
        k_multi_head = k_multi_head.transpose(1,2)
        v_multi_head = v_multi_head.transpose(1,2)
        
        # 3. 计算注意力分数 [batch_size, n_head, seq_len, seq_len]
        # 转置k为 [batch_size, n_head, head_dim, seq_len]
        k_t = k_multi_head.transpose(2,3)
        
        # 批量矩阵乘法计算注意力分数
        attention_scores = q_multi_head @ k_t / math.sqrt(head_dim)
        
        # 修改后代码
        if past_kv is not None:
            # 创建一个掩码张量
            mask_value = torch.finfo(torch.float32).min
            # 创建掩码并应用 - 上三角为True(掩盖部分)
            mask = torch.triu(torch.ones((seq_len, k.shape[1])), diagonal=1) > 0
            mask = mask[-seq_len:, :]
            mask = mask.reshape(1, 1, seq_len, k.shape[1])
            mask = torch.tensor(mask, dtype=torch.bool)
            # 应用掩码 (在softmax之前)
            attention_scores.masked_fill_(mask, mask_value)
            # 然后应用softmax
            attention_weights = F.softmax(attention_scores, dim=-1)
        else:
            # 创建一个掩码张量
            mask_value = torch.finfo(torch.float32).min
            # 修改：使用 > 0 确保上三角为True(需要掩盖的部分)
            mask = torch.triu(torch.ones((seq_len, seq_len)), diagonal=1) > 0
            mask = mask.reshape(1, 1, seq_len, seq_len)
            mask = torch.tensor(mask, dtype=torch.bool)
            attention_scores.masked_fill_(mask, mask_value)
            # 然后应用softmax
            attention_weights = F.softmax(attention_scores, dim=-1)

        
        if self.enable_monitoring and self.monitor:
            self.monitor.record(f"layer_{layer_idx}_attention_weights", attention_weights)

        attention_weights = layer.attn.attn_dropout(attention_weights) 
        # 6. 应用注意力权重到V
        context = attention_weights@v_multi_head

        # 7. 转置回原始形状，然后合并头部 [batch_size, seq_len, hidden_size]
        context = context.transpose(1,2).reshape(batch_size, seq_len, hidden_size)
        context_flat = context.reshape(-1, hidden_size)
        # 在计算投影之前，添加软件计算投影
        if self.enable_monitoring and self.monitor:
            # 获取投影权重和偏置
            proj_weight = layer.attn.c_proj.weight
            proj_bias = layer.attn.c_proj.bias if hasattr(layer.attn.c_proj, 'bias') and layer.attn.c_proj.bias is not None else None
            
            # 软件计算投影
            attn_output_software = context_flat@ proj_weight
            if proj_bias is not None:
                attn_output_software = attn_output_software + proj_bias
            
            # 记录软件计算结果
            self.monitor.record(f"layer_{layer_idx}_attn_output_software", 
                            attn_output_software.reshape(batch_size, seq_len, -1))
        
        # 计算投影 (硬件)    
        attention_output=self._hardware_processing(f'L{layer_idx}_Wproj',context_flat,'注意力输出投影').reshape(batch_size, seq_len, self.config.n_embd)
        attention_output = layer.attn.resid_dropout(attention_output)
 
        # 添加投影偏置
        if hasattr(layer.attn.c_proj, 'bias') and layer.attn.c_proj.bias is not None:
            attention_output += layer.attn.c_proj.bias
            self.monitor.record(f"layer_{layer_idx}_attn_output_bias", layer.attn.c_proj.bias)
        # 监测注意力输出
        if self.enable_monitoring and self.monitor:
            self.monitor.record(f"layer_{layer_idx}_attn_weights", attention_weights)
            self.monitor.record(f"layer_{layer_idx}_attn_output_hardware", attention_output)
        
        if use_cache:
            return attention_output, present_kv
        return attention_output,None
    
    def forward_mlp(self, hidden_states: torch.tensor, layer_idx):
        """
        使用硬件模拟器执行MLP计算
        
        参数:
        - hidden_states: 输入隐藏状态
        - layer_idx: 层索引
        
        返回:
        - mlp_output: MLP层的输出
        """
        # 监测输入
        if self.enable_monitoring and self.monitor:
            self.monitor.record(f"layer_{layer_idx}_mlp_input", hidden_states)
        
        # 获取当前层的GPT-2实现，用于访问某些参数和偏置值
        layer = self.model.transformer.h[layer_idx]
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # 为硬件模拟器准备输入
        hidden_flat = hidden_states.reshape(-1, hidden_size)
        
        # 软件计算第一个全连接层
        if self.enable_monitoring and self.monitor:
            # 获取fc1权重和偏置
            fc1_weight = layer.mlp.c_fc.weight
            fc1_bias = layer.mlp.c_fc.bias if hasattr(layer.mlp.c_fc, 'bias') and layer.mlp.c_fc.bias is not None else None
            
            # 软件计算fc1输出
            fc1_output_software = torch.matmul(hidden_flat, fc1_weight)
            if fc1_bias is not None:
                fc1_output_software = fc1_output_software + fc1_bias
            
            # 应用GELU激活函数
            fc1_activated_software = F.gelu(fc1_output_software)
            
            # 记录软件计算结果
            self.monitor.record(f"layer_{layer_idx}_mlp_fc1_output_software", 
                            fc1_output_software.reshape(batch_size, seq_len, -1))
            self.monitor.record(f"layer_{layer_idx}_mlp_activated_software", 
                            fc1_activated_software.reshape(batch_size, seq_len, -1))
        
        # 计算第一个全连接层 (硬件)
        hidden1=self._hardware_processing(f"L{layer_idx}_Wfc1", hidden_flat, "MLP第一层").reshape(batch_size, seq_len, self.config.n_inner)
        
        # 添加偏置和应用激活函数
        if hasattr(layer.mlp.c_fc, 'bias') and layer.mlp.c_fc.bias is not None:
            hidden1 += layer.mlp.c_fc.bias
            self.monitor.record(f"layer_{layer_idx}_mlp_fc1_output_bias", layer.mlp.c_fc.bias)
        
        # 应用GELU激活函数（使用软件计算）
        hidden1_activated = F.gelu(hidden1)
        
        # 监测FC1结果
        if self.enable_monitoring and self.monitor:
            self.monitor.record(f"layer_{layer_idx}_mlp_fc1_output_hardware", hidden1)
            self.monitor.record(f"layer_{layer_idx}_mlp_activated_hardware", hidden1_activated)
        
        # 软件计算第二个全连接层
        if self.enable_monitoring and self.monitor:
            # 获取fc2权重和偏置
            fc2_weight = layer.mlp.c_proj.weight
            fc2_bias = layer.mlp.c_proj.bias if hasattr(layer.mlp.c_proj, 'bias') and layer.mlp.c_proj.bias is not None else None
            
            # 软件计算fc2输出
            hidden1_activated_tensor = hidden1_activated.reshape(-1, self.config.n_inner)
            fc2_output_software = torch.matmul(hidden1_activated_tensor, fc2_weight)
            if fc2_bias is not None:
                fc2_output_software = fc2_output_software + fc2_bias
            
            # 记录软件计算结果
            self.monitor.record(f"layer_{layer_idx}_mlp_output_software", 
                            fc2_output_software.reshape(batch_size, seq_len, -1))
        

        # 计算第二个全连接层 (硬件)
        hidden1_activated_flat = hidden1_activated.reshape(-1, self.config.n_inner)      
        mlp_output=self._hardware_processing(f"L{layer_idx}_Wfc2", hidden1_activated_flat, "MLP第二层").reshape(batch_size, seq_len, hidden_size)
        mlp_output = layer.mlp.dropout(mlp_output)
        
        # 添加偏置
        if hasattr(layer.mlp.c_proj, 'bias') and layer.mlp.c_proj.bias is not None:
            mlp_output += layer.mlp.c_proj.bias
            self.monitor.record(f"layer_{layer_idx}_mlp_output_bias", layer.mlp.c_proj.bias)  
            # 监测MLP输出
        if self.enable_monitoring and self.monitor:
            self.monitor.record(f"layer_{layer_idx}_mlp_output_hardware", mlp_output)    
        return mlp_output
        
    def forward(self, input_ids:torch.tensor, past_key_values=None, use_cache=True, position_ids=None):
        """
        执行GPT-2前向计算
        
        参数:
        - input_ids: 输入token IDs
        - past_key_values: 过去的key-value缓存
        - use_cache: 是否使用和返回缓存
        - position_ids: 位置编码IDs，如果为None则自动生成
        
        返回:
        - logits: 输出logits
        - present_key_values: 当前的key-value缓存(如果use_cache=True)
        """
        batch_size, seq_len = input_ids.shape
        # 嵌入层计算（使用软件计算）
        # 词嵌入 50257 * 768
        input_embeds = self.model.transformer.wte.weight[input_ids]
        
        # 位置嵌入 1024的最大长度
        if position_ids is None:
            # 如果没有提供position_ids，则自动生成
            if past_key_values is not None:
                # 对于有KV缓存的情况，位置从past_length开始
                past_length = past_key_values[0][0].shape[2] if past_key_values[0][0].ndim > 2 else 0
                position_ids = torch.arange(past_length, past_length + seq_len).reshape(1, -1).repeat(batch_size,1)
            else:
                # 没有KV缓存时，从0开始
                position_ids = torch.arange(0, seq_len).reshape(1, -1).repeat(batch_size,1)
        
        position_embeds = self.model.transformer.wpe.weight[position_ids]
        
        # 合并嵌入
        hidden_states = input_embeds + position_embeds
        hidden_states = self.model.transformer.drop(hidden_states)
        # 准备保存key-value缓存的列表
        presents = [] if use_cache else None
        
        # 处理每一层
        for i, block in enumerate(self.model.transformer.h):
            # 提取过去的KV缓存（如果提供）
            self.logger.info(f'开始处理第{i}层')

            layer_past = past_key_values[i] if past_key_values is not None else None
            
            # 第一个层归一化
            ln1_output = block.ln_1(hidden_states)
            
            # 自注意力计算（硬件加速）
            if use_cache:
                attn_output, present = self.forward_attention(ln1_output, i, past_kv=layer_past, use_cache=True)
                presents.append(present)
            else:
                attn_output = self.forward_attention(ln1_output, i, past_kv=None, use_cache=False)
            
            # 残差连接（使用软件计算）
            hidden_states = hidden_states + attn_output

            # 第二个层归一化
            ln2_output = block.ln_2(hidden_states) 
            
            # MLP计算（硬件加速）
            mlp_output = self.forward_mlp(ln2_output, i)
            
            # 残差连接（使用软件计算）
            hidden_states = hidden_states + mlp_output
            
        # 最后的层归一化
        hidden_states = self.model.transformer.ln_f(hidden_states)
        logits = self.model.lm_head(hidden_states)
        return logits, presents if use_cache else None
 
    def generate(self, prompt, max_length, temperature, top_k, device='cpu'):
        """生成文本"""
        # 记录生成开始时间
        start_time = time.time()
        if self.enable_monitoring and self.monitor:
            self.logger.info("模型监测已启用，将记录中间计算结果")

        # 对输入文本进行tokenize
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        
        # 保存KV缓存
        past_key_values = None

        generated_text = prompt
        self.logger.info("\n开始生成文本:")
        print(f"{prompt}", end="", flush=True)
        
        # 限制最大长度
        max_length = min(max_length, self.model.config.n_positions)
        # 生成过程
        for step in range(max_length):
            self.logger.debug(f"开始生成步骤{step}的文本")
            # 设置当前步骤
            if self.enable_monitoring:
                self.monitor.set_step(step)
                
            # 优化：根据past_key_values裁剪输入
            if past_key_values is not None:
                # 只使用最后一个token作为输入
                input_ids_current = input_ids[:, -1:].clone()
                # 计算正确的位置编码
                past_length = past_key_values[0][0].shape[1] if past_key_values[0][0].ndim > 2 else 0
                position_ids = [[past_length]]
            else:
                input_ids_current = input_ids
                position_ids = None
            
            # 执行模型前向计算
            with torch.no_grad():
                outputs = self.forward(
                    input_ids_current, 
                    past_key_values=past_key_values, 
                    use_cache=True,
                    position_ids=position_ids
                )
                logits, past_key_values = outputs
                logits = logits[:, -1, :]  # 只保留最后一个token的logits
                
            # 应用temperature
            logits = logits / temperature
            
            # 将logits转换为概率
            probs = F.softmax(logits, dim=-1)
            
            # 使用top-k sampling
            top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
            
            # 正则化top-k概率
            top_k_probs = top_k_probs / top_k_probs.sum()
            
            # 从top-k中采样
            next_token_id = top_k_indices[0, torch.multinomial(top_k_probs[0], 1).item()].item()
            
            # 保存当前token的概率分布
            next_token_text = self.tokenizer.decode([next_token_id])
            # 监测token概率
            if self.enable_monitoring and self.monitor:
                self.monitor.plot_token_probabilities(probs[0], next_token_text, next_token_id, self.tokenizer)
                
            # 将新token添加到输入序列
            input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]]).to(device)], dim=1)
            
            # 解码并打印新token
            token_text = self.tokenizer.decode([next_token_id])
            generated_text += token_text
            print(token_text, end="", flush=True)
            
            # 如果生成了终止标记，则停止
            if next_token_id == self.tokenizer.eos_token_id:
                break
        self.total_power+=self.hardware_sim.bg_power*self.total_time #背景电流功耗        
        self.logger.info(f'计算背景电流能耗，耗能{self.hardware_sim.bg_power*self.total_time}uJ，目前总耗能{self.total_power}uJ') 
        # 解码生成的tokens
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        # 计算生成时间
        generation_time = time.time() - start_time
        self.logger.info(f"生成文本：{generated_text}")
        self.logger.info(f"文本生成完成，耗时：{generation_time:.2f}秒，硬件计算耗时：{self.total_time/1e3:.2f}毫秒")
        
        # 如果启用监测，执行最终分析
        if self.enable_monitoring and self.monitor:
            self.logger.info("生成完成，正在分析模型计算过程...")
            self.monitor.compare_original_vs_quantized()
            self.monitor.save_computation_records()
        return generated_text, self.total_time
            
