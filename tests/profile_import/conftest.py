"""Prevent the parent suite's destructive database-cleanup fixture in this folder."""

import pytest


@pytest.fixture(autouse=True)
def clean_database():
    """Import tests create their own in-memory database; never clean the app database."""
