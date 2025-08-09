# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 15:43:59 2023

@author: Martin-PC
"""

import tensorflow as tf
print(f"Tensorflow version: {tf.__version__}")
print(f"Keras Version: {tf.keras.__version__}")
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")
