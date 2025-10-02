import pandas as pd

class DataHandler:
    def __init__(self):
        ...
    
    @staticmethod
    def add_weekday_column(hourly_detail: pd.DataFrame):
        """
        DataFrame에서 날짜 컬럼을 기준으로 요일 컬럼을 추가하는 함수
        
        Parameters:
            prep ("PreProcessing"): 입력 데이터프레임을 포함한 인스턴스.
        """
        # 요일 컬럼 추가 (한글로 표시하려면 day_name() 대신 weekday() 활용 가능)
        hourly_detail['날짜'] = pd.to_datetime(hourly_detail['날짜'])
        hourly_detail["요일"] = hourly_detail['날짜'].dt.day_name(locale='ko_KR.utf8')
        return hourly_detail