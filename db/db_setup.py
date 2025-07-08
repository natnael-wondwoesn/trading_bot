# SQLite Setup (Development)
import aiosqlite


async def init_db():
    async with aiosqlite.connect("trading_bot.db") as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                symbol TEXT,
                side TEXT,
                amount REAL,
                price REAL,
                pnl REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        await db.commit()
