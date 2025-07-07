import sys
import os
import io
import subprocess
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, 
                            QGroupBox, QFormLayout, QTabWidget, QTextEdit, QFileDialog,
                            QCheckBox, QLineEdit, QMessageBox, QScrollArea)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import config
# 导入各个模块
import main
import error_analyze
import power_analyze
from time_analyze import GPT2TimeAnalyzer

class OutputCapture:
    """用于捕获print输出的类"""
    def __init__(self, output_callback):
        self.output_callback = output_callback
        self.buffer = io.StringIO()
        
    def write(self, text):
        self.buffer.write(text)
        self.output_callback(text)
        
    def flush(self):
        pass
    
    def get_output(self):
        return self.buffer.getvalue()


class GPT2SimulationGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GPT-2 硬件仿真工具")
        self.resize(1000, 800)
        
        # 创建主部件和布局
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # 创建选项卡
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)
        
        # 初始化任务队列
        self.task_queue = []
        
        # 添加选项卡页面
        self.setup_config_tab()
        self.setup_simulation_tab()
        self.setup_analysis_tab()
        
        # 添加底部按钮
        self.setup_buttons()
        
    def setup_config_tab(self):
        """设置配置选项卡"""
        config_tab = QWidget()
        config_layout = QVBoxLayout(config_tab)
        
        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        config_layout.addWidget(scroll_area)
        
        # 创建滚动内容窗口
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_area.setWidget(scroll_content)
        
        # 硬件配置组
        hardware_group = QGroupBox("硬件配置")
        hardware_layout = QFormLayout()
        scroll_layout.addWidget(hardware_group)
        hardware_group.setLayout(hardware_layout)
        
        # 量化位数设置
        self.quant_bits = QComboBox()
        self.quant_bits.addItems(["4", "6", "8", "12", "16","32"])
        self.quant_bits.setCurrentText(str(config.QUANTIZATION_BITS))
        hardware_layout.addRow("量化位数 (QUANTIZATION_BITS):", self.quant_bits)
        
        # BL大小
        self.bl_size = QSpinBox()
        self.bl_size.setMinimum(1)
        self.bl_size.setMaximum(131072)
        self.bl_size.setValue(config.BITLINES_PER_PLANE)
        hardware_layout.addRow("Bitline大小 (BITLINES_PER_PLANE):", self.bl_size)
        
        # Block大小
        self.block_size = QSpinBox()
        self.block_size.setMinimum(1)
        self.block_size.setMaximum(300)
        self.block_size.setValue(config.BLOCKS_PER_PLANE)
        hardware_layout.addRow("Block大小 (BLOCKS_PER_PLANE):", self.block_size)
        
        # 对称量化设置
        self.symmetric = QCheckBox()
        self.symmetric.setChecked(config.SYMMETRIC)
        hardware_layout.addRow("对称量化 (SYMMETRIC):", self.symmetric)
        
        # 电流设置
        self.current = QSpinBox()
        self.current.setMinimum(1)
        self.current.setMaximum(1000)
        self.current.setValue(config.MAX_CURRENT)
        hardware_layout.addRow("电流值 (MAX_CURRENT):", self.current)
        
        # 电流变化模式
        self.current_variation_mode = QCheckBox()
        self.current_variation_mode.setChecked(config.CURRENT_VARIATION_MODE)
        hardware_layout.addRow("电流分布误差 (CURRENT_VARIATION_MODE):", self.current_variation_mode)
        
        # 每个操作的BL数
        self.max_current_sum = QSpinBox()
        self.max_current_sum.setMinimum(1)
        self.max_current_sum.setMaximum(500)
        self.max_current_sum.setValue(config.MAX_CURRENT_SUM)
        hardware_layout.addRow("每个操作的BL数 (MAX_CURRENT_SUM):", self.max_current_sum)
        
        # 每个操作的块数
        self.blocks_per_operation = QSpinBox()
        self.blocks_per_operation.setMinimum(1)
        self.blocks_per_operation.setMaximum(300)
        self.blocks_per_operation.setValue(config.BLOCKS_PER_OPERATION)
        hardware_layout.addRow("每个操作的块数 (BLOCKS_PER_OPERATION):", self.blocks_per_operation)
        
        # 硬件架构参数组
        arch_group = QGroupBox("硬件架构参数")
        arch_layout = QFormLayout()
        scroll_layout.addWidget(arch_group)
        arch_group.setLayout(arch_layout)
        
        # ADC复用次数
        self.num_tia_times_adc = QSpinBox()
        self.num_tia_times_adc.setMinimum(1)
        self.num_tia_times_adc.setMaximum(1000)
        self.num_tia_times_adc.setValue(config.NUM_TIA_TIMES_ADC)
        arch_layout.addRow("ADC复用次数 (NUM_TIA_TIMES_ADC):", self.num_tia_times_adc)
        
        # 每个die的平面数
        self.planes_per_die = QSpinBox()
        self.planes_per_die.setMinimum(1)
        self.planes_per_die.setMaximum(8)
        self.planes_per_die.setValue(config.PLANES_PER_DIE)
        arch_layout.addRow("每个die的平面数 (PLANES_PER_DIE):", self.planes_per_die)
        
        # 每个die的层数
        self.layer_per_die = QSpinBox()
        self.layer_per_die.setMinimum(1)
        self.layer_per_die.setMaximum(96)
        self.layer_per_die.setValue(config.LAYER_PER_DIE)
        arch_layout.addRow("每个die的层数 (LAYER_PER_DIE):", self.layer_per_die)
        
        # 每个块的TSG数
        self.tsg_per_block = QSpinBox()
        self.tsg_per_block.setMinimum(1)
        self.tsg_per_block.setMaximum(10)
        self.tsg_per_block.setValue(config.TSG_PER_BLOCK)
        arch_layout.addRow("每个块的TSG数 (TSG_PER_BLOCK):", self.tsg_per_block)
        
        # Y方向损失
        self.y_lost = QSpinBox()
        self.y_lost.setMinimum(0)
        self.y_lost.setMaximum(10)
        self.y_lost.setValue(config.Y_LOST)
        arch_layout.addRow("Y方向损失 (Y_LOST):", self.y_lost)
        
        # TIA数量基础值
        self.tia_num_base = QSpinBox()
        self.tia_num_base.setMinimum(1)
        self.tia_num_base.setMaximum(40000)
        self.tia_num_base.setValue(config.TIA_NUM_BASE)
        arch_layout.addRow("TIA数量基础值 (TIA_NUM_BASE):", self.tia_num_base)
        
        # 硬件时序参数组
        timing_group = QGroupBox("硬件时序参数")
        timing_layout = QFormLayout()
        scroll_layout.addWidget(timing_group)
        timing_group.setLayout(timing_layout)
        
        # WL设置时间基础值
        self.wl_setup_time_base = QDoubleSpinBox()
        self.wl_setup_time_base.setDecimals(2)
        self.wl_setup_time_base.setMinimum(0.01)
        self.wl_setup_time_base.setMaximum(19)
        self.wl_setup_time_base.setValue(config.WL_SETUP_TIME_BASE)
        timing_layout.addRow("WL设置时间基础值 (WL_SETUP_TIME_BASE):", self.wl_setup_time_base)
        
        # TSG切换时间
        self.tsg_switch_time = QDoubleSpinBox()
        self.tsg_switch_time.setDecimals(2)
        self.tsg_switch_time.setMinimum(0.01)
        self.tsg_switch_time.setMaximum(19)
        self.tsg_switch_time.setValue(config.TSG_SWITCH_TIME)
        timing_layout.addRow("TSG设置时间 (TSG_SWITCH_TIME):", self.tsg_switch_time)

        # WL切换时间
        self.wl_switch_time = QDoubleSpinBox()
        self.wl_switch_time.setDecimals(2)
        self.wl_switch_time.setMinimum(0.01)
        self.wl_switch_time.setMaximum(19)
        self.wl_switch_time.setValue(config.WL_SWITCH_TIME)
        timing_layout.addRow("WL设置时间 (WL_SWITCH_TIME):", self.wl_switch_time)        
        
        # BL切换时间
        self.bl_switch_time = QDoubleSpinBox()
        self.bl_switch_time.setDecimals(2)
        self.bl_switch_time.setMinimum(0.01)
        self.bl_switch_time.setMaximum(20)
        self.bl_switch_time.setValue(config.BL_SWITCH_TIME)
        timing_layout.addRow("BL设置时间 (BL_SWITCH_TIME):", self.bl_switch_time)
        
        # TIA转换时间
        self.tia_conversion_time = QDoubleSpinBox()
        self.tia_conversion_time.setDecimals(3)
        self.tia_conversion_time.setMinimum(1e-3)
        self.tia_conversion_time.setMaximum(10.0)
        self.tia_conversion_time.setValue(config.TIA_CONVERSION_TIME)
        timing_layout.addRow("TIA转换时间 (TIA_CONVERSION_TIME):", self.tia_conversion_time)
        
        # ADC转换时间基础值
        self.adc_conversion_time_base = QDoubleSpinBox()
        self.adc_conversion_time_base.setDecimals(3)
        self.adc_conversion_time_base.setMinimum(1e-3)
        self.adc_conversion_time_base.setMaximum(1.0)
        self.adc_conversion_time_base.setValue(config.ADC_CONVERSION_TIME_BASE)
        timing_layout.addRow("ADC转换时间基础值 (ADC_CONVERSION_TIME_BASE):", self.adc_conversion_time_base)
        
        # 写入时间
        self.write_time = QDoubleSpinBox()
        self.write_time.setDecimals(1)
        self.write_time.setMinimum(1.0)
        self.write_time.setMaximum(1000.0)
        self.write_time.setValue(config.WRITE_TIME)
        timing_layout.addRow("写入时间 (WRITE_TIME):", self.write_time)
        
        # 电力参数组
        power_group = QGroupBox("电力参数")
        power_layout = QFormLayout()
        scroll_layout.addWidget(power_group)
        power_group.setLayout(power_layout)
        
        # TSG和WL电流
        self.tsg_wl_current = QDoubleSpinBox()
        self.tsg_wl_current.setDecimals(5)
        self.tsg_wl_current.setMinimum(1e-9)
        self.tsg_wl_current.setMaximum(1.0)
        self.tsg_wl_current.setValue(config.TSG_WL_CURRENT)
        power_layout.addRow("TSG和WL电流 (TSG_WL_CURRENT):", self.tsg_wl_current)
        
        # BL电流
        self.bl_current = QDoubleSpinBox()
        self.bl_current.setDecimals(9)
        self.bl_current.setMinimum(1e-9)
        self.bl_current.setMaximum(1.0)
        self.bl_current.setValue(config.BL_CURRENT)
        power_layout.addRow("BL电流 (BL_CURRENT):", self.bl_current)
        
        # 电源电压
        self.vcc = QDoubleSpinBox()
        self.vcc.setDecimals(2)
        self.vcc.setMinimum(0.1)
        self.vcc.setMaximum(10.0)
        self.vcc.setValue(config.VCC)
        power_layout.addRow("电源电压 (VCC):", self.vcc)
        
        # TIA和ADC功耗基础值
        self.tia_adc_power_base = QDoubleSpinBox()
        self.tia_adc_power_base.setDecimals(6)
        self.tia_adc_power_base.setMinimum(1e-6)
        self.tia_adc_power_base.setMaximum(1.0)
        self.tia_adc_power_base.setValue(config.TIA_ADC_POWER_BASE)
        power_layout.addRow("TIA和ADC功耗基础值 (TIA_ADC_POWER_BASE):", self.tia_adc_power_base)
        
        # 权重配置组
        weights_group = QGroupBox("权重配置")
        weights_layout = QFormLayout()
        scroll_layout.addWidget(weights_group)
        weights_group.setLayout(weights_layout)
        
        # 权重均值
        self.weight_mean = QDoubleSpinBox()
        self.weight_mean.setDecimals(3)
        self.weight_mean.setMinimum(-10.0)
        self.weight_mean.setMaximum(10.0)
        self.weight_mean.setValue(config.MEAN)
        weights_layout.addRow("权重均值 (MEAN):", self.weight_mean)
        
        # 权重缩放
        self.weight_scale = QDoubleSpinBox()
        self.weight_scale.setDecimals(6)
        self.weight_scale.setMinimum(0.000001)
        self.weight_scale.setMaximum(10.0)
        self.weight_scale.setValue(config.SCALE)
        weights_layout.addRow("权重缩放 (SCALE):", self.weight_scale)
        
        # 模型配置组
        model_group = QGroupBox("模型配置")
        model_layout = QFormLayout()
        scroll_layout.addWidget(model_group)
        model_group.setLayout(model_layout)
        
        # 启用监控
        self.enable_monitoring = QCheckBox()
        self.enable_monitoring.setChecked(config.ENABLE_MONITORING)
        model_layout.addRow("启用监控 (ENABLE_MONITORING):", self.enable_monitoring)
        
        # 可视化设置
        self.visualize = QCheckBox()
        self.visualize.setChecked(config.VISUALIZE)
        model_layout.addRow("可视化 (VISUALIZE):", self.visualize)
        
        # 生成配置组
        generation_group = QGroupBox("生成配置")
        generation_layout = QFormLayout()
        scroll_layout.addWidget(generation_group)
        generation_group.setLayout(generation_layout)
        
        # 温度参数
        self.temperature = QDoubleSpinBox()
        self.temperature.setDecimals(2)
        self.temperature.setMinimum(0.1)
        self.temperature.setMaximum(2.0)
        self.temperature.setValue(config.DEFAULT_TEMPERATURE)
        generation_layout.addRow("温度参数 (DEFAULT_TEMPERATURE):", self.temperature)
        
        # Top-K参数
        self.top_k = QSpinBox()
        self.top_k.setMinimum(1)
        self.top_k.setMaximum(100)
        self.top_k.setValue(config.DEFAULT_TOP_K)
        generation_layout.addRow("Top-K参数 (DEFAULT_TOP_K):", self.top_k)
        
        # 最大生成长度
        self.max_length = QSpinBox()
        self.max_length.setMinimum(1)
        self.max_length.setMaximum(1024)
        self.max_length.setValue(config.DEFAULT_MAX_LENGTH)
        generation_layout.addRow("最大生成长度 (DEFAULT_MAX_LENGTH):", self.max_length)
        
        # 提示词
        self.prompt = QLineEdit()
        self.prompt.setText(config.PROMPT)
        generation_layout.addRow("提示词 (PROMPT):", self.prompt)
        
        # 输出配置组
        output_group = QGroupBox("输出配置")
        output_layout = QFormLayout()
        scroll_layout.addWidget(output_group)
        output_group.setLayout(output_layout)
        
        # 输出目录
        self.output_dir = QLineEdit()
        self.output_dir.setText(config.CONFIG_PATH)
        output_layout.addRow("输出目录 (CONFIG_PATH):", self.output_dir)
        browse_button = QPushButton("浏览...")
        browse_button.clicked.connect(self.browse_output_dir)
        output_layout.addRow("", browse_button)
        
        # 添加任务队列状态显示
        queue_group = QGroupBox("任务队列状态")
        queue_layout = QVBoxLayout()
        queue_group.setLayout(queue_layout)
        scroll_layout.addWidget(queue_group)
        
        self.queue_status_label = QLabel("当前队列: 0 个任务")
        queue_layout.addWidget(self.queue_status_label)
        
        # 添加队列详情显示
        self.queue_details = QTextEdit()
        self.queue_details.setReadOnly(True)
        self.queue_details.setMaximumHeight(100)
        queue_layout.addWidget(QLabel("队列任务:"))
        queue_layout.addWidget(self.queue_details)
        
        # 添加清空队列按钮
        self.clear_queue_button = QPushButton("清空队列")
        self.clear_queue_button.clicked.connect(self.clear_queue)
        queue_layout.addWidget(self.clear_queue_button)
        
        # 添加配置选项卡
        self.tabs.addTab(config_tab, "配置设置")
    def validate_config(self):
        """验证配置变量之间的约束关系"""
        # 用于存储错误消息的列表
        error_messages = []
        
        # 检查y_lost和layer_per_die之间的约束关系
        if self.y_lost.value() >= self.bl_size.value():
            error_messages.append("Y方向损失(Y_LOST)必须小于每个Plane的BL数(BITLINES_PER_PLANE)")
        
        # 检查块大小与平面大小的关系
        if self.blocks_per_operation.value() > self.block_size.value():
            error_messages.append("每个操作的块数(BLOCKS_PER_OPERATION)不能大于每个平面的块数(BLOCKS_PER_PLANE)")
        
        # 检查BL相关参数
        if self.max_current_sum.value() > self.bl_size.value():
            error_messages.append("每个操作的BL数(MAX_CURRENT_SUM)不能大于Bitline大小(BITLINES_PER_PLANE)")
        
          
        # 如果有错误消息，显示错误对话框并返回False
        if error_messages:
            QMessageBox.critical(self, "配置错误", "\n".join(error_messages))
            return False
        
        # 所有约束都满足，返回True
        return True        
    def setup_simulation_tab(self):
        """设置仿真选项卡"""
        simulation_tab = QWidget()
        layout = QVBoxLayout(simulation_tab)
        
        # 仿真日志
        self.simulation_log = QTextEdit()
        self.simulation_log.setReadOnly(True)
        layout.addWidget(QLabel("仿真日志:"))
        layout.addWidget(self.simulation_log)
        
        # 添加仿真选项卡
        self.tabs.addTab(simulation_tab, "仿真结果")
        
    def setup_analysis_tab(self):
        """设置分析选项卡"""
        analysis_tab = QWidget()
        layout = QVBoxLayout(analysis_tab)
        
        # 分析选项
        analysis_options = QGroupBox("分析选项")
        analysis_layout = QVBoxLayout()
        analysis_options.setLayout(analysis_layout)
        layout.addWidget(analysis_options)
        
        # 时间分析参数
        time_group = QGroupBox("时间分析")
        time_layout = QFormLayout()
        time_group.setLayout(time_layout)
        analysis_layout.addWidget(time_group)
        
        self.time_log_file = QLineEdit()
        browse_time_button = QPushButton("选择日志文件")
        browse_time_button.clicked.connect(lambda: self.browse_log_file(self.time_log_file))
        time_layout.addRow("日志文件:", self.time_log_file)
        time_layout.addRow("", browse_time_button)
        
        # 功耗分析参数
        power_group = QGroupBox("功耗分析")
        power_layout = QFormLayout()
        power_group.setLayout(power_layout)
        analysis_layout.addWidget(power_group)
        
        self.power_log_file = QLineEdit()
        browse_power_button = QPushButton("选择日志文件")
        browse_power_button.clicked.connect(lambda: self.browse_log_file(self.power_log_file))
        power_layout.addRow("日志文件:", self.power_log_file)
        power_layout.addRow("", browse_power_button)
        
        self.start_step = QSpinBox()
        self.start_step.setMinimum(0)
        self.start_step.setMaximum(999)
        self.start_step.setSpecialValueText("开始")
        power_layout.addRow("起始步骤:", self.start_step)
        
        self.end_step = QSpinBox()
        self.end_step.setMinimum(0)
        self.end_step.setMaximum(999)
        self.end_step.setSpecialValueText("结束")
        power_layout.addRow("结束步骤:", self.end_step)
        
        # 误差分析参数
        error_group = QGroupBox("误差分析")
        error_layout = QFormLayout()
        error_group.setLayout(error_layout)
        analysis_layout.addWidget(error_group)
        
        self.file1_path = QLineEdit()
        browse_file1_button = QPushButton("选择基准文件")
        browse_file1_button.clicked.connect(lambda: self.browse_log_file(self.file1_path, "选择基准metrics文件"))
        error_layout.addRow("基准文件:", self.file1_path)
        error_layout.addRow("", browse_file1_button)
        
        self.file2_path = QLineEdit()
        browse_file2_button = QPushButton("选择对比文件")
        browse_file2_button.clicked.connect(lambda: self.browse_log_file(self.file2_path, "选择对比metrics文件"))
        error_layout.addRow("对比文件:", self.file2_path)
        error_layout.addRow("", browse_file2_button)
        
        self.target_layers = QLineEdit()
        self.target_layers.setText("layer_0_mlp_fc2,layer_1_mlp_fc2,layer_2_mlp_fc2")
        error_layout.addRow("目标层(逗号分隔):", self.target_layers)
        
        # 分析日志
        self.analysis_log = QTextEdit()
        self.analysis_log.setReadOnly(True)
        layout.addWidget(QLabel("分析日志:"))
        layout.addWidget(self.analysis_log)
        
        # 添加分析选项卡
        self.tabs.addTab(analysis_tab, "分析工具")
        
    def setup_buttons(self):
        """设置底部按钮"""
        button_layout = QHBoxLayout()
        self.main_layout.addLayout(button_layout)
        
        # 应用配置按钮
        self.apply_config_button = QPushButton("应用配置")
        self.apply_config_button.clicked.connect(self.apply_config)
        button_layout.addWidget(self.apply_config_button)
        
        # 添加到任务队列按钮
        self.add_queue_button = QPushButton("添加到任务队列")
        self.add_queue_button.clicked.connect(self.add_to_queue)
        button_layout.addWidget(self.add_queue_button)
        
        # 启动仿真按钮
        self.run_simulation_button = QPushButton("启动仿真")
        self.run_simulation_button.clicked.connect(self.run_simulation)
        button_layout.addWidget(self.run_simulation_button)
        
        # 时间分析按钮
        self.time_analysis_button = QPushButton("执行时间分析")
        self.time_analysis_button.clicked.connect(self.run_time_analysis)
        button_layout.addWidget(self.time_analysis_button)
        
        # 功耗分析按钮
        self.power_analysis_button = QPushButton("执行功耗分析")
        self.power_analysis_button.clicked.connect(self.run_power_analysis)
        button_layout.addWidget(self.power_analysis_button)
        
        # 误差分析按钮
        self.error_analysis_button = QPushButton("执行误差分析")
        self.error_analysis_button.clicked.connect(self.run_error_analysis)
        button_layout.addWidget(self.error_analysis_button)
        
    def apply_config(self):
        """应用配置到config.py文件"""
        # 首先验证配置
        if not self.validate_config():
            return

        try:
            # 读取现有的config.py文件
            with open("config.py", "r", encoding="utf-8") as f:
                config_lines = f.readlines()
                
            # 准备新的配置值
            new_config = {
                "QUANTIZATION_BITS": int(self.quant_bits.currentText()),
                "BITLINES_PER_PLANE": self.bl_size.value(),
                "BLOCKS_PER_PLANE": self.block_size.value(),
                "SYMMETRIC": self.symmetric.isChecked(),
                "MAX_CURRENT": self.current.value(),
                "CURRENT_VARIATION_MODE": self.current_variation_mode.isChecked(),
                "MAX_CURRENT_SUM": self.max_current_sum.value(),
                "BLOCKS_PER_OPERATION": self.blocks_per_operation.value(),
                "NUM_TIA_TIMES_ADC": self.num_tia_times_adc.value(),
                "PLANES_PER_DIE": self.planes_per_die.value(),
                "LAYER_PER_DIE": self.layer_per_die.value(),
                "TSG_PER_BLOCK": self.tsg_per_block.value(),
                "Y_LOST": self.y_lost.value(),
                "TIA_NUM_BASE": self.tia_num_base.value(),
                "WL_SETUP_TIME_BASE": self.wl_setup_time_base.value(),
                "TSG_SWITCH_TIME": self.tsg_switch_time.value(), 
                "WL_SWITCH_TIME": self.wl_switch_time.value(),    
                "BL_SWITCH_TIME": self.bl_switch_time.value(),
                "TIA_CONVERSION_TIME": self.tia_conversion_time.value(),
                "ADC_CONVERSION_TIME_BASE": self.adc_conversion_time_base.value(),
                "WRITE_TIME": self.write_time.value(),
                "TSG_WL_CURRENT": self.tsg_wl_current.value(),
                "BL_CURRENT": self.bl_current.value(),
                "VCC": self.vcc.value(),
                "TIA_ADC_POWER_BASE": self.tia_adc_power_base.value(),
                "MEAN": self.weight_mean.value(),
                "SCALE": self.weight_scale.value(),
                "ENABLE_MONITORING": self.enable_monitoring.isChecked(),
                "VISUALIZE": self.visualize.isChecked(),
                "DEFAULT_TEMPERATURE": self.temperature.value(),
                "DEFAULT_TOP_K": self.top_k.value(),
                "DEFAULT_MAX_LENGTH": self.max_length.value(),
                "PROMPT": self.prompt.text(),
                "CONFIG_PATH": self.output_dir.text()
            }
            
            # 更新配置文件
            new_config_lines = []
            for line in config_lines:
                for key, value in new_config.items():
                    if line.strip().startswith(key + " =") or line.strip().startswith(key + "="):
                        if isinstance(value, bool):
                            line = f"{key} = {str(value)}\n"
                        elif isinstance(value, str):
                            line = f'{key} = "{value}"\n'
                        else:
                            line = f"{key} = {value}\n"
                        break
                new_config_lines.append(line)
            
            # 写入新配置
            with open("config.py", "w", encoding="utf-8") as f:
                f.writelines(new_config_lines)
                
            QMessageBox.information(self, "成功", "配置已成功应用")
            
            # 更新模块中的配置
            import importlib
            importlib.reload(config)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"应用配置时出错: {str(e)}")
    
    def add_to_queue(self):
        """将当前配置添加到任务队列"""
        if not self.validate_config():
            return
        # 获取当前配置
        current_config = {
            "QUANTIZATION_BITS": int(self.quant_bits.currentText()),
            "BITLINES_PER_PLANE": self.bl_size.value(),
            "BLOCKS_PER_PLANE": self.block_size.value(),
            "SYMMETRIC": self.symmetric.isChecked(),
            "MAX_CURRENT": self.current.value(),
            "CURRENT_VARIATION_MODE": self.current_variation_mode.isChecked(),
            "MAX_CURRENT_SUM": self.max_current_sum.value(),
            "BLOCKS_PER_OPERATION": self.blocks_per_operation.value(),
            "NUM_TIA_TIMES_ADC": self.num_tia_times_adc.value(),
            "PLANES_PER_DIE": self.planes_per_die.value(),
            "LAYER_PER_DIE": self.layer_per_die.value(),
            "TSG_PER_BLOCK": self.tsg_per_block.value(),
            "Y_LOST": self.y_lost.value(),
            "TIA_NUM_BASE": self.tia_num_base.value(),
            "WL_SETUP_TIME_BASE": self.wl_setup_time_base.value(),
            "TSG_SWITCH_TIME": self.tsg_switch_time.value(),  
            "WL_SWITCH_TIME": self.wl_switch_time.value(),    
            "BL_SWITCH_TIME": self.bl_switch_time.value(),
            "TIA_CONVERSION_TIME": self.tia_conversion_time.value(),
            "ADC_CONVERSION_TIME_BASE": self.adc_conversion_time_base.value(),
            "WRITE_TIME": self.write_time.value(),
            "TSG_WL_CURRENT": self.tsg_wl_current.value(),
            "BL_CURRENT": self.bl_current.value(),
            "VCC": self.vcc.value(),
            "TIA_ADC_POWER_BASE": self.tia_adc_power_base.value(),
            "MEAN": self.weight_mean.value(),
            "SCALE": self.weight_scale.value(),
            "ENABLE_MONITORING": self.enable_monitoring.isChecked(),
            "VISUALIZE": self.visualize.isChecked(),
            "DEFAULT_TEMPERATURE": self.temperature.value(),
            "DEFAULT_TOP_K": self.top_k.value(),
            "DEFAULT_MAX_LENGTH": self.max_length.value(),
            "PROMPT": self.prompt.text(),
            "CONFIG_PATH": self.output_dir.text()
        }
        
        # 添加到队列
        self.task_queue.append(current_config)
        
        # 更新队列状态显示
        self.queue_status_label.setText(f"当前队列: {len(self.task_queue)} 个任务")
        
        # 更新队列详情
        self.update_queue_details()
        
        QMessageBox.information(self, "成功", f"已将当前配置添加到任务队列\n队列中共有 {len(self.task_queue)} 个任务")
    
    def clear_queue(self):
        """清空任务队列"""
        self.task_queue = []
        self.queue_status_label.setText("当前队列: 0 个任务")
        self.queue_details.clear()
        QMessageBox.information(self, "成功", "已清空任务队列")
    
    def update_queue_details(self):
        """更新队列详情显示"""
        details = ""
        for i, config in enumerate(self.task_queue):
            details += f"任务 {i+1}: 量化位数={config['QUANTIZATION_BITS']}, "
            details += f"BL大小={config['BITLINES_PER_PLANE']}, "
            details += f"Block大小={config['BLOCKS_PER_PLANE']}\n"
        
        self.queue_details.setText(details)
        
    def browse_output_dir(self):
        """浏览选择输出目录"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if dir_path:
            self.output_dir.setText(dir_path)
    
    def browse_log_file(self, line_edit, title="选择日志文件"):
        """浏览选择日志文件"""
        file_path, _ = QFileDialog.getOpenFileName(self, title, "", "日志文件 (*.log *.txt);;所有文件 (*.*)")
        if file_path:
            line_edit.setText(file_path)
    
    def update_simulation_log(self, text):
        """更新仿真日志"""
        self.simulation_log.append(text)
        # 滚动到底部
        self.simulation_log.verticalScrollBar().setValue(
            self.simulation_log.verticalScrollBar().maximum()
        )
    
    def update_analysis_log(self, text):
        """更新分析日志"""
        self.analysis_log.append(text)
        # 滚动到底部
        self.analysis_log.verticalScrollBar().setValue(
            self.analysis_log.verticalScrollBar().maximum()
        )
    
    def run_simulation(self):
        """直接运行仿真"""
        # 检查是否有任务队列
        if self.task_queue:
            # 清空日志
            self.simulation_log.clear()
            self.simulation_log.append(f"正在启动仿真队列 (共 {len(self.task_queue)} 个任务)...\n")
            
            # 设置按钮状态
            self.run_simulation_button.setEnabled(False)
            
            # 捕获标准输出
            output_capture = OutputCapture(self.update_simulation_log)
            old_stdout = sys.stdout
            sys.stdout = output_capture
            
            try:
                # 直接运行main模块的主函数，传入配置列表
                main_result = main.main(config_list=self.task_queue)
                
                # 恢复标准输出
                sys.stdout = old_stdout
                
                # 清空队列
                self.task_queue = []
                self.queue_status_label.setText("当前队列: 0 个任务")
                self.queue_details.clear()
                
                self.update_simulation_log("\n所有任务仿真完成！")
                QMessageBox.information(self, "完成", "所有任务仿真执行完成")
            # except Exception as e:
            #     # 恢复标准输出
            #     sys.stdout = old_stdout
                
            #     error_msg = f"仿真队列执行出错: {str(e)}"
            #     self.update_simulation_log(f"\n{error_msg}")
            #     QMessageBox.critical(self, "错误", f"仿真队列执行失败: {str(e)}")
            finally:
                # 恢复按钮状态
                self.run_simulation_button.setEnabled(True)
        else:
            # 确保配置已应用
            self.apply_config()
            
            # 清空日志
            self.simulation_log.clear()
            self.simulation_log.append("正在启动仿真...\n")
            
            # 设置按钮状态
            self.run_simulation_button.setEnabled(False)
            
            # 捕获标准输出
            output_capture = OutputCapture(self.update_simulation_log)
            old_stdout = sys.stdout
            sys.stdout = output_capture
            
            try:
                # 直接运行main模块的主函数
                main_result = main.main()  # 使用当前配置
                
                # 恢复标准输出
                sys.stdout = old_stdout
                
                self.update_simulation_log("\n仿真完成！")
                QMessageBox.information(self, "完成", "仿真执行完成")
            except Exception as e:
                # 恢复标准输出
                sys.stdout = old_stdout
                
                error_msg = f"仿真出错: {str(e)}"
                self.update_simulation_log(f"\n{error_msg}")
                QMessageBox.critical(self, "错误", f"仿真执行失败: {str(e)}")
            finally:
                # 恢复按钮状态
                self.run_simulation_button.setEnabled(True)
    
    def run_time_analysis(self):
        """直接运行时间分析"""
        # 清空分析日志
        self.analysis_log.clear()
        self.analysis_log.append("开始执行时间分析...\n")
        
        # 捕获标准输出
        output_capture = OutputCapture(self.update_analysis_log)
        old_stdout = sys.stdout
        sys.stdout = output_capture
        
        try:
            # 检查是否选择了日志文件
            log_file = self.time_log_file.text().strip()
            if not log_file:
                raise ValueError("请选择要分析的日志文件")
            
            if not os.path.exists(log_file):
                raise ValueError(f"文件不存在: {log_file}")
            
            # 创建分析器并分析
            analyzer = GPT2TimeAnalyzer(log_file)
            output_dir = "time_analysis"
            os.makedirs(output_dir, exist_ok=True)
            analyzer.analyze_all(output_dir=output_dir)
            
            # 恢复标准输出
            sys.stdout = old_stdout
            
            self.update_analysis_log("\n时间分析完成！")
            self.show_analysis_results(output_dir, "时间")
            
        except Exception as e:
            # 恢复标准输出
            sys.stdout = old_stdout
            
            error_msg = f"时间分析出错: {str(e)}"
            self.update_analysis_log(f"\n{error_msg}")
            QMessageBox.critical(self, "错误", f"时间分析执行失败: {str(e)}")
    
    def run_power_analysis(self):
        """直接运行功耗分析"""
        # 清空分析日志
        self.analysis_log.clear()
        self.analysis_log.append("开始执行功耗分析...\n")
        
        # 捕获标准输出
        output_capture = OutputCapture(self.update_analysis_log)
        old_stdout = sys.stdout
        sys.stdout = output_capture
        
        try:
            # 检查是否选择了日志文件
            log_file = self.power_log_file.text().strip()
            if not log_file:
                raise ValueError("请选择要分析的日志文件")
            
            if not os.path.exists(log_file):
                raise ValueError(f"文件不存在: {log_file}")
            
            # 获取步骤范围
            start_step = None if self.start_step.value() == 0 and self.start_step.specialValueText() == "开始" else self.start_step.value()
            end_step = None if self.end_step.value() == 0 and self.end_step.specialValueText() == "结束" else self.end_step.value()
            
            # 创建分析器并分析
            analyzer = power_analyze.GPT2PowerAnalyzer(log_file, start_step, end_step)
            output_dir = "power_analysis"
            os.makedirs(output_dir, exist_ok=True)
            analyzer.analyze_all(output_dir=output_dir)
            
            # 恢复标准输出
            sys.stdout = old_stdout
            
            self.update_analysis_log("\n功耗分析完成！")
            self.show_analysis_results(output_dir, "功耗")
            
        except Exception as e:
            # 恢复标准输出
            sys.stdout = old_stdout
            
            error_msg = f"功耗分析出错: {str(e)}"
            self.update_analysis_log(f"\n{error_msg}")
            QMessageBox.critical(self, "错误", f"功耗分析执行失败: {str(e)}")
    
    def run_error_analysis(self):
        """直接运行误差分析"""
        # 清空分析日志
        self.analysis_log.clear()
        self.analysis_log.append("开始执行误差分析...\n")
        
        # 捕获标准输出
        output_capture = OutputCapture(self.update_analysis_log)
        old_stdout = sys.stdout
        sys.stdout = output_capture
        
        try:
            # 检查是否选择了两个文件
            file1 = self.file1_path.text().strip()
            file2 = self.file2_path.text().strip()
            
            if not file1 or not file2:
                raise ValueError("请选择要比较的两个metrics文件")
            
            if not os.path.exists(file1) or not os.path.exists(file2):
                raise ValueError("所选文件不存在")
            
            # 获取目标层
            target_layers_str = self.target_layers.text().strip()
            if not target_layers_str:
                raise ValueError("请指定要分析的目标层")
            
            target_layers = [layer.strip() for layer in target_layers_str.split(",")]
            
            # 创建输出目录
            output_dir = "error_analysis"
            os.makedirs(output_dir, exist_ok=True)
            
            # 执行比较
            comparison = error_analyze.compare_metrics(file1, file2, target_layers, ["mae", "rmse", "relative_error"])
            
            # 保存数据到Excel
            error_analyze.save_data_to_excel(comparison, output_dir)
            
            # 可视化比较结果
            error_analyze.visualize_comparison(comparison, ["mae", "rmse", "relative_error"], output_dir)
            
            # 恢复标准输出
            sys.stdout = old_stdout
            
            self.update_analysis_log("\n误差分析完成！")
            self.show_analysis_results(output_dir, "误差")
            
        except Exception as e:
            # 恢复标准输出
            sys.stdout = old_stdout
            
            error_msg = f"误差分析出错: {str(e)}"
            self.update_analysis_log(f"\n{error_msg}")
            QMessageBox.critical(self, "错误", f"误差分析执行失败: {str(e)}")
    
    def show_analysis_results(self, result_dir, analysis_type):
        """显示分析结果"""
        if os.path.exists(result_dir):
            # 打开结果文件夹
            if sys.platform == 'win32':
                os.startfile(result_dir)
            elif sys.platform == 'darwin':  # macOS
                subprocess.run(['open', result_dir])
            else:  # Linux
                subprocess.run(['xdg-open', result_dir])
                
            QMessageBox.information(self, "完成", f"{analysis_type}分析执行完成，结果已保存在 {result_dir} 目录")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GPT2SimulationGUI()
    window.show()
    sys.exit(app.exec())