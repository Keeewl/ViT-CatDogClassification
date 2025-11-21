import sys
import os

# 获取当前脚本的绝对路径，再向上跳转一级（回到项目根目录）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import numpy as np
import os
