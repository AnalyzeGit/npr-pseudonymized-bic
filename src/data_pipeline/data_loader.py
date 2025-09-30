import pandas as pd
from typing import Tuple

class DataLoader:
    def __init__(self, usage_path: str, info_path: str):
        self.usage_path = usage_path
        self.info_path = info_path

    def __call__(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        CSV 파일에서 usage 데이터와 info 데이터를 불러와
        인스턴스 속성에 저장하고 반환합니다.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                - usage (pd.DataFrame): usage 데이터
                - info (pd.DataFrame): info 데이터
        """

        usage = pd.read_csv(self.usage_path)
        info = pd.read_csv(self.info_path)

        return usage, info 



        