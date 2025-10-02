from .data_prep import PreProcessing
from ..utils.path_config import PROCESSED_DATA_DIR

class ResultReporter:
    def __init__(self, prep: "PreProcessing"):
        self.prep = prep

    def print_day_counts(self) -> None:
            """최근 분석 결과에서 일수 요약을 콘솔에 출력"""
            if self.prep.last_result is None:
                raise RuntimeError("먼저 analyze()를 실행하세요.")

            dc = self.prep.last_result.get('day_counts')
            dcd = self.prep.last_result.get('day_counts_by_dow')

            if dc is None or dcd is None:
                raise RuntimeError("일수 테이블이 없습니다. analyze()가 최신 코드로 실행되었는지 확인하세요.")

            total = int(dc.loc[dc['구분'] == '전체', '일수'].iat[0])
            weekday = int(dc.loc[dc['구분'] == '평일', '일수'].iat[0])
            weekend = int(dc.loc[dc['구분'] == '주말', '일수'].iat[0])

            print("▶ 분석 일수 요약")
            print(f" - 총 분석일수: {total}일 (평일 {weekday}일, 주말 {weekend}일)")
            print("\n▶ 요일별 일수")
            print(dcd.to_string(index=False))


    def export_results(
        self,
        folder_name, 
        encoding: str = 'cp949',
    ) -> None:
        """마지막 분석 결과를 CSV로 저장 (평일/주말, 일수 테이블 포함)"""
        if self.prep.last_result is None:
            raise RuntimeError("먼저 analyze()를 실행하세요.")

        for key, df in self.prep.last_result.items():
            file = PROCESSED_DATA_DIR / folder_name / f"{key}.csv"
            df.to_csv(file, index=False, encoding=encoding)