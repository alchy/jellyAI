"""
JellyAIview API Test Client
Created: 2025-02-06 09:39:43 UTC
Author: alchy
"""

import requests
import random
import logging
from typing import Dict
import string

# Nastavení loggeru
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class JellyAPI:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.api_endpoint = f"{base_url}/api/data"
        logger.info(f"Initialized API test client for {self.base_url}")

    def _generate_random_label(self, prefix: str = "Test") -> str:
        """Generuje náhodný label."""
        random_suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
        return f"{prefix}_{random_suffix}"

    def _generate_random_value(self, min_val: float = -1.0, max_val: float = 1.0) -> float:
        """Generuje náhodnou hodnotu v rozsahu [-1.0, 1.0]."""
        return round(random.uniform(min_val, max_val), 3)

    def generate_test_data(self, count: int = 5) -> Dict[str, float]:
        """
        Generuje testovací data ve formátu:
        {
            "label1": value1,
            "label2": value2,
            ...
        }
        """
        return {
            self._generate_random_label(): self._generate_random_value()
            for _ in range(count)
        }

    def add_data(self, data: Dict[str, float]) -> bool:
        """Odesílá data na API."""
        try:
            logger.debug(f"Sending data: {data}")
            response = requests.post(self.api_endpoint, json=data)

            if response.status_code == 422:
                logger.error(f"Invalid data format. Response: {response.json()}")
                return False

            response.raise_for_status()
            logger.info(f"Successfully sent {len(data)} items")

            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send data: {e}")
            return False

    def get_current_data(self) -> Dict[str, float]:
        """Získá aktuální data z API."""
        try:
            response = requests.get(self.api_endpoint)
            response.raise_for_status()
            data = response.json()["data"]

            print(f"\nCurrent data list [{len(data)} objects]:")
            for i, obj in enumerate(data):
                print(f"[{i}]: {obj}")

            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get data: {e}")
            return {}

    def clear_data(self) -> bool:
        """Vymaže všechna data."""
        try:
            response = requests.delete(self.api_endpoint)
            response.raise_for_status()
            logger.info("Successfully cleared all data")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to clear data: {e}")
            return False