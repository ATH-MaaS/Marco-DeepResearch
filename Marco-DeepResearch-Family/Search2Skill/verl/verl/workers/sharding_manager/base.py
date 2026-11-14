"""
Sharding manager to implement HybridEngine
"""

from verl import DataProto

class BaseShardingManager:
    def __init__(self):
        self.timing = {}

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def preprocess_data(self, data: DataProto) -> DataProto:
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        return data
