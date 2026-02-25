"""
Main entry point for Real-Time Face Swap Application
"""
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from ui import MainWindow
from utils import logger
import config

def main():
    """Main application entry point"""
    try:
        # Create Qt application
        app = QApplication(sys.argv)
        app.setApplicationName(config.APP_NAME)
        
        # Create and show main window
        window = MainWindow()
        window.show()
        
        logger.info(f"{config.APP_NAME} v{config.VERSION} started")
        
        # Run application
        sys.exit(app.exec_())
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()