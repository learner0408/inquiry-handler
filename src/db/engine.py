"""Shared SQLAlchemy Base + async/sync engine helpers (psycopg3)."""

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from src.config import get_settings


class Base(DeclarativeBase):
    pass


def get_sync_engine():  # used by scripts/init_db.py
    return create_engine(get_settings().DATABASE_URL, pool_size=5)


def get_async_engine():  # used by FastAPI handlers
    return create_async_engine(get_settings().DATABASE_URL, pool_size=5, max_overflow=5)


AsyncSessionFactory = async_sessionmaker(get_async_engine(), expire_on_commit=False)
