"""model 包：时序建模模块（兼容原 volatility 目录）。

按优化要求新增 model/ 目录，模块从 volatility/ 复用导出，
避免全盘重构代码，同时满足仓库目录规范。
"""

from volatility.realized_vol import RealizedVolatility
from volatility.garch import GARCHModel
from volatility.arima_model import ARIMAModel
from volatility.lstm_vol import LSTMVolatility
from volatility.transformer_model import AttentionTransformer
from volatility.highfreq import HighFreqData

__all__ = [
    "RealizedVolatility",
    "GARCHModel",
    "ARIMAModel",
    "LSTMVolatility",
    "AttentionTransformer",
    "HighFreqData",
]
