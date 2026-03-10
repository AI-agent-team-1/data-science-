from bot_plaxin import bot


if __name__ == "__main__":
    print("Бот запущен...")
    bot.infinity_polling(timeout=10, long_polling_timeout=5)

