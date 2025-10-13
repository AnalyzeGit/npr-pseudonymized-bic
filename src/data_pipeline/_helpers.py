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
    
    @staticmethod
    def classify_dominant_day(row: pd.Series) -> str:
        start = row['입차일시']
        end = row['출차일시']
        
        # 1시간 단위로 분할
        hours = pd.date_range(start, end, freq='H')
        
        # 평일(0~4), 주말(5~6)
        weekday_hours = sum(h.weekday() < 5 for h in hours)
        weekend_hours = len(hours) - weekday_hours
        
        # 분류
        if weekday_hours > weekend_hours:
            return '평일'
        elif weekend_hours > weekday_hours:
            return '주말'
        else:
            return '중립'  # (예: 정확히 반반인 경우)
        
    @staticmethod
    def remove_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna(subset='출차일시')