import pandas as pd
from .path_config import PROCESSED_DATA_DIR

class ResultExport:
    def __init__(self, result: pd.DataFrame):
        self.result = result

    def __call__(self, file_name: str, encoding: str = 'cp949') -> None:
        """마지막 분석 결과를 CSV로 저장 (평일/주말, 일수 테이블 포함)"""
        if self.result is None:
            raise RuntimeError("데이터를 인풋하세요.")

        file = PROCESSED_DATA_DIR  / file_name
        self.result.to_csv(file, index=False, encoding=encoding)