"""
Run Telegram Bot

This script runs the Telegram bot for punter predictions.
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def main():
    """Run the Telegram bot."""
    # Check if Telegram bot token is set
    telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")

    if not telegram_bot_token:
        logger.error("TELEGRAM_BOT_TOKEN environment variable not set. Please set it and try again.")
        sys.exit(1)

    # Check if Telegram group ID is set
    telegram_group_id = os.getenv("TELEGRAM_GROUP_ID")

    if not telegram_group_id:
        logger.warning("TELEGRAM_GROUP_ID environment variable not set. Bot will listen to all messages.")

    # Import and run the bot
    try:
        # Add the current directory to the Python path
        sys.path.append(os.getcwd())

        # Import the telegram_bot module
        from telegram_bot import main as run_bot

        logger.info("Starting Telegram bot...")
        run_bot()
    except ImportError as e:
        logger.error(f"Could not import telegram_bot module: {str(e)}")
        logger.error("Make sure it exists and python-telegram-bot is installed.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error running Telegram bot: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
