#!/usr/bin/env python3
"""
Start BetSightly Backend with Telegram Bot

This script starts both the main FastAPI application and the Telegram bot.
"""

import os
import sys
import time
import signal
import logging
import subprocess
import threading
from pathlib import Path

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class ServiceManager:
    """Manages both the main app and Telegram bot."""
    
    def __init__(self):
        self.main_app_process = None
        self.telegram_bot_process = None
        self.running = True
        
    def start_main_app(self):
        """Start the main FastAPI application."""
        try:
            logger.info("🚀 Starting main FastAPI application...")
            
            # Get port from environment
            port = os.getenv("PORT", "8000")
            
            # Start gunicorn
            cmd = [
                "gunicorn", 
                "main:app",
                "--bind", f"0.0.0.0:{port}",
                "--workers", "1",
                "--worker-class", "uvicorn.workers.UvicornWorker"
            ]
            
            self.main_app_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            logger.info(f"✅ Main app started on port {port}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start main app: {e}")
            return False
    
    def start_telegram_bot(self):
        """Start the Telegram bot."""
        try:
            # Check if Telegram bot token is set
            if not os.getenv("TELEGRAM_BOT_TOKEN"):
                logger.warning("⚠️  TELEGRAM_BOT_TOKEN not set - skipping Telegram bot")
                return True
            
            # Check if telegram_bot.py exists
            if not Path("telegram_bot.py").exists():
                logger.warning("⚠️  telegram_bot.py not found - skipping Telegram bot")
                return True
            
            logger.info("🤖 Starting Telegram bot...")
            
            # Start Telegram bot
            self.telegram_bot_process = subprocess.Popen(
                [sys.executable, "telegram_bot.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            logger.info("✅ Telegram bot started")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start Telegram bot: {e}")
            return False
    
    def monitor_processes(self):
        """Monitor both processes and restart if needed."""
        while self.running:
            try:
                # Check main app
                if self.main_app_process and self.main_app_process.poll() is not None:
                    logger.error("❌ Main app process died - restarting...")
                    self.start_main_app()
                
                # Check Telegram bot
                if self.telegram_bot_process and self.telegram_bot_process.poll() is not None:
                    logger.warning("⚠️  Telegram bot process died - restarting...")
                    self.start_telegram_bot()
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"❌ Error in process monitoring: {e}")
                time.sleep(5)
    
    def stop_all(self):
        """Stop all processes."""
        logger.info("🛑 Stopping all services...")
        self.running = False
        
        if self.main_app_process:
            try:
                self.main_app_process.terminate()
                self.main_app_process.wait(timeout=10)
                logger.info("✅ Main app stopped")
            except Exception as e:
                logger.error(f"❌ Error stopping main app: {e}")
                try:
                    self.main_app_process.kill()
                except:
                    pass
        
        if self.telegram_bot_process:
            try:
                self.telegram_bot_process.terminate()
                self.telegram_bot_process.wait(timeout=10)
                logger.info("✅ Telegram bot stopped")
            except Exception as e:
                logger.error(f"❌ Error stopping Telegram bot: {e}")
                try:
                    self.telegram_bot_process.kill()
                except:
                    pass

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"📡 Received signal {signum}")
    if hasattr(signal_handler, 'service_manager'):
        signal_handler.service_manager.stop_all()
    sys.exit(0)

def main():
    """Main function."""
    logger.info("🚀 Starting BetSightly Backend with Telegram Bot")
    
    # Create service manager
    service_manager = ServiceManager()
    signal_handler.service_manager = service_manager
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start main app
        if not service_manager.start_main_app():
            logger.error("❌ Failed to start main app")
            return False
        
        # Start Telegram bot
        service_manager.start_telegram_bot()
        
        # Start monitoring in a separate thread
        monitor_thread = threading.Thread(target=service_manager.monitor_processes)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        logger.info("✅ All services started successfully")
        logger.info("📋 Services running:")
        logger.info("   - Main FastAPI app")
        if os.getenv("TELEGRAM_BOT_TOKEN"):
            logger.info("   - Telegram bot")
        logger.info("")
        logger.info("🌐 API Endpoints:")
        logger.info("   - Health: /api/health")
        logger.info("   - Predictions: /api/predictions/")
        logger.info("   - Betting Codes: /api/betting-codes/")
        logger.info("   - Punters: /api/punters/")
        logger.info("")
        logger.info("🤖 Telegram Bot:")
        if os.getenv("TELEGRAM_BOT_TOKEN"):
            logger.info("   - Status: Running")
            logger.info("   - Add bot to your group and send:")
            logger.info("     Code: ABC123")
            logger.info("     Odds: 1.85")
            logger.info("     Bookmaker: Bet365")
        else:
            logger.info("   - Status: Disabled (no TELEGRAM_BOT_TOKEN)")
        
        # Keep main thread alive
        while service_manager.running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("📡 Received keyboard interrupt")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
    finally:
        service_manager.stop_all()
    
    logger.info("👋 Shutdown complete")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
