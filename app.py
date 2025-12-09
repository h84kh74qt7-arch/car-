import sys
import os

# Add the current directory to path
sys.path.append(os.path.dirname(__file__))

from ML.car.gradio_app import demo

if __name__ == "__main__":
    demo.launch()
