import numpy as np
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGridLayout, QPushButton, QScrollArea, QSizePolicy,
    QHBoxLayout, QLineEdit, QComboBox
)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib

# 配置matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 避免负号显示为方块

class HardwareVisualizer(QWidget):
    """
    3D NAND存储单元可视化工具，用于交互式查看和调试存储阵列
    """
    
    def __init__(self, cell_array):
        """
        初始化可视化器
        
        参数:
        - cell_array: 存储单元阵列数据
        """
        super().__init__()
        self.cell_array = cell_array
        self.initUI()
        self.current_layer = None
        self.current_plane = None
        self.current_tsg = None
        self.current_canvas = None
        self.current_nav_layout = None

    def initUI(self):
        """初始化用户界面"""
        self.setWindowTitle("3D NAND Hardware Visualizer")
        self.setGeometry(100, 100, 800, 600)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # 添加导航标签
        self.nav_label = QLabel("存储层级导航")
        self.nav_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.nav_label)

        self.scroll_area = QScrollArea()
        self.scroll_area_widget = QWidget()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.scroll_area_widget)
        self.scroll_layout = QVBoxLayout(self.scroll_area_widget)
        self.layout.addWidget(self.scroll_area)

        self.show_layer_view()  # 初始显示层级选择

    def clear_dynamic_layout(self):
        """清空动态布局"""
        def clear_layout(layout):
            if layout is not None:
                while layout.count():
                    item = layout.takeAt(0)
                    child_widget = item.widget()
                    if child_widget is not None:
                        child_widget.deleteLater()
                    else:
                        child_layout = item.layout()
                        if child_layout is not None:
                            clear_layout(child_layout)
                            
        clear_layout(self.scroll_layout)
        self.current_canvas = None
        self.current_nav_layout = None

    def add_back_button(self, target):
        """添加返回按钮"""
        back_button = QPushButton("返回")
        back_button.clicked.connect(target)
        back_button.setStyleSheet("background-color: #2196F3; color: white;")
        self.scroll_layout.addWidget(back_button)

    def show_layer_view(self):
        """显示Layer层级选择"""
        self.clear_dynamic_layout()
        self.nav_label.setText("选择 Layer")

        grid = QGridLayout()
        layers = self.cell_array.shape[1]  # 获取实际层数
        
        # 动态计算布局行列（示例：每行3个）
        cols = 3
        rows = (layers + cols - 1) // cols
        for i in range(rows):
            for j in range(cols):
                layer_idx = i * cols + j
                if layer_idx >= layers:
                    break
                btn = QPushButton(f"Layer {layer_idx}")
                btn.clicked.connect(lambda _, l=layer_idx: self.show_plane_view(l))
                btn.setStyleSheet("font-size: 14px; padding: 15px;")
                grid.addWidget(btn, i, j)
        
        self.scroll_layout.addLayout(grid)

    def show_plane_view(self, layer):
        """显示Plane层级选择"""
        self.clear_dynamic_layout()
        self.current_layer = layer
        self.nav_label.setText(f"Layer {layer} → 选择 Plane")

        grid = QGridLayout()
        planes = self.cell_array.shape[0]  # 获取实际Plane数量

        # 动态计算布局行列
        cols = 3
        rows = (planes + cols - 1) // cols
        for i in range(rows):
            for j in range(cols):
                plane_idx = i * cols + j
                if plane_idx >= planes:
                    break
                btn = QPushButton(f"Plane {plane_idx}")
                btn.clicked.connect(lambda _, p=plane_idx: self.show_tsg_view(layer, p))
                btn.setStyleSheet("font-size: 12px; padding: 10px;")
                grid.addWidget(btn, i, j)

        self.scroll_layout.addLayout(grid)
        self.add_back_button(target=self.show_layer_view)

    def show_tsg_view(self, layer, plane):
        """显示TSG层级选择，跳过Block层级"""
        self.clear_dynamic_layout()
        self.current_plane = plane
        self.nav_label.setText(f"Layer {layer} → Plane {plane} → 选择 TSG")

        grid = QGridLayout()
        tsgs = self.cell_array.shape[3]  # 获取实际TSG数量

        # 动态计算布局行列
        cols = 5
        rows = (tsgs + cols - 1) // cols
        for i in range(rows):
            for j in range(cols):
                tsg_idx = i * cols + j
                if tsg_idx >= tsgs:
                    break
                btn = QPushButton(f"TSG {tsg_idx}")
                btn.clicked.connect(lambda _, t=tsg_idx: self.show_cell_view(layer, plane, t))
                btn.setStyleSheet("font-size: 12px; padding: 8px;")
                grid.addWidget(btn, i, j)

        self.scroll_layout.addLayout(grid)
        self.add_back_button(target=lambda: self.show_plane_view(layer))

    def show_cell_view(self, layer, plane, tsg):
        """显示Cell存储状态，展示所有Block"""
        self.clear_dynamic_layout()
        self.current_tsg = tsg
        title = f"Layer {layer} → Plane {plane} → TSG {tsg}"
        self.nav_label.setText(title + " → Cell View")

        # 范围选择控件
        range_layout = QHBoxLayout()
        self.bl_start_input = QLineEdit()
        self.bl_start_input.setPlaceholderText("起始 BL (默认: 0)")
        self.bl_start_input.setText("0")
        range_layout.addWidget(QLabel("BL 起始位置:"))
        range_layout.addWidget(self.bl_start_input)
        
        # 显示按钮
        show_btn = QPushButton("渲染所有 Blocks 的 Cells")
        show_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        show_btn.clicked.connect(lambda: self.display_cell_data(layer, plane, tsg))
        
        self.scroll_layout.addLayout(range_layout)
        self.scroll_layout.addWidget(show_btn)
        self.add_back_button(target=lambda: self.show_tsg_view(layer, plane))

    def display_cell_data(self, layer, plane, tsg):
        """显示所有Block的128根BL数据，形成矩阵"""
        # 获取BL起始位置
        try:
            bl_start = int(self.bl_start_input.text() or 0)
        except ValueError:
            error_label = QLabel("输入无效！请输入数字。")
            self.scroll_layout.addWidget(error_label)
            return

        # 限制合法范围
        max_bl_idx = self.cell_array.shape[4] - 129  # 预留128个BL的空间
        bl_start = max(0, min(bl_start, max_bl_idx))
        bl_end = bl_start + 127  # 每页显示128根BL
        
        # 创建矩阵存储所有Block的当前BL数据
        blocks_data = np.zeros((self.cell_array.shape[2], 128), dtype=np.bool_)
        
        # 获取每个Block对应BL位置的数据
        for block_idx in range(self.cell_array.shape[2]):
            blocks_data[block_idx, :] = self.cell_array[
                plane,      # Plane维度
                layer,      # Layer维度
                block_idx,  # 遍历所有Block
                tsg,        # TSG维度
                bl_start:bl_end+1  # 128根BL
            ]
        
        # 移除之前的画布和导航控件
        if self.current_canvas is not None:
            self.scroll_layout.removeWidget(self.current_canvas)
            self.current_canvas.deleteLater()
            self.current_canvas = None
            
        if self.current_nav_layout is not None:
            self.scroll_layout.removeItem(self.current_nav_layout)
            while self.current_nav_layout.count():
                item = self.current_nav_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
        
        # 创建新画布
        plt.close('all')  # 关闭所有之前的图形
        self.figure, self.ax = plt.subplots(figsize=(12, 10))
        self.current_canvas = FigureCanvas(self.figure)
        self.scroll_layout.addWidget(self.current_canvas)
        
        # 添加BL位置导航控件
        bl_nav_layout = QHBoxLayout()
        prev_btn = QPushButton("上一页 BL")
        next_btn = QPushButton("下一页 BL")
        bl_position_label = QLabel(f"当前显示: BL {bl_start} - {bl_end}")
        
        prev_btn.clicked.connect(lambda: self.change_bl_position(layer, plane, tsg, bl_start - 128))
        next_btn.clicked.connect(lambda: self.change_bl_position(layer, plane, tsg, bl_start + 128))
        
        bl_nav_layout.addWidget(prev_btn)
        bl_nav_layout.addWidget(bl_position_label)
        bl_nav_layout.addWidget(next_btn)
        self.scroll_layout.addLayout(bl_nav_layout)
        self.current_nav_layout = bl_nav_layout  # 保存对导航布局的引用
        
        # 渲染矩阵
        self.render_cell_matrix(blocks_data, bl_start, bl_end)
        
    def change_bl_position(self, layer, plane, tsg, new_bl_start):
        """改变BL位置并重新渲染"""
        max_bl_idx = self.cell_array.shape[4] - 129
        new_bl_start = max(0, min(new_bl_start, max_bl_idx))
        self.bl_start_input.setText(str(new_bl_start))
        self.display_cell_data(layer, plane, tsg)
    
    def render_cell_matrix(self, blocks_data, bl_start, bl_end):
        """渲染矩阵"""
        # 清空当前绘图
        self.ax.clear()
        
        # 绘制热力图
        self.ax.imshow(
            blocks_data,
            cmap='Greys',  # 使用Greys颜色映射
            aspect='auto',  # 自动调整纵横比
            interpolation='none'  # 禁用插值
        )
        
        # 添加网格线
        self.ax.set_xticks(np.arange(-0.5, blocks_data.shape[1], 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, blocks_data.shape[0], 1), minor=True)
        self.ax.grid(which="minor", color="gray", linestyle='--', linewidth=0.5)
        
        # 隐藏主网格线
        self.ax.tick_params(which="major", bottom=False, left=False)
        
        # 设置坐标标签
        self.ax.set_xlabel(f"Bit Line Index ({bl_start} - {bl_end})")
        self.ax.set_ylabel("Block Index (0 - 215)")
        self.ax.set_title(f"所有Block的Cell状态 (BL {bl_start}-{bl_end})")
        
        # 调整布局避免冲突
        self.figure.tight_layout()
        
        # 刷新画布
        self.current_canvas.draw()
