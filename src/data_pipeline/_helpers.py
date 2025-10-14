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
    

    @staticmethod
    # (3). 정기권 + 주간 정기권 + 야간 정기권 -> 정기권 / 나머지 비정기권
    def calculate_parking_duration(df: pd.DataFrame) -> pd.DataFrame:
        # 문자열을 datetime 타입으로 변환
        df['입차일시'] = pd.to_datetime(df['입차일시'])
        df['출차일시'] = pd.to_datetime(df['출차일시'])
        
        # 이용시간(분) 계산
        df['이용시간(분)'] = (df['출차일시'] - df['입차일시']).dt.total_seconds() / 60

        # 이용시간(시간)으로도 추가
        df['이용시간(시간)'] = df['이용시간(분)'] / 60
        return df
    

    @staticmethod
    def flag_regular_customers(df: pd.DataFrame) -> pd.DataFrame:
        df['is_regular'] = False
        df.loc[df['입차유형'].isin(['정기권', '주간정기권', '야간정기권']), 'is_regular'] = True
        return df
    
    @staticmethod
    def aggregate_by_hour(df: pd.DataFrame) -> pd.DataFrame:
        # === 핵심 로직: 시간대별 집계 ===
        usage = []
        for h in range(24):
            count = ((df["입차일시"].dt.hour*60 + df["입차일시"].dt.minute < h*60+60) &
                    (df["출차일시"].dt.hour*60 + df["출차일시"].dt.minute >= h*60)).sum()
        
            usage.append({"hour": h, "users": count})
        
        usage_df = pd.DataFrame(usage)
        return usage_df
    
    
    @staticmethod
    def avg_daily_entries_by_hour(df: pd.DataFrame) -> pd.DataFrame:
        """
        여러 날짜 데이터에서
        시간대별 하루 평균 입차 차량 수(회전율)를 계산하는 함수
        """

        # 1️⃣ 날짜·시간 처리
        df['입차일시'] = pd.to_datetime(df['입차일시'])
        df['입차일'] = df['입차일시'].dt.date          # 입차일 (일자 단위)
        df['입차_시'] = df['입차일시'].dt.hour

        # 2️⃣ 각 날짜(day)·시간대(hour)별 입차 대수
        daily_counts = df.groupby(['입차일', '입차_시']).size().reset_index(name='입차대수')

        # 3️⃣ 시간대별 평균 (즉, 하루 평균 입차량)
        avg_per_hour = (
            daily_counts.groupby('입차_시')['입차대수']
            .mean()
            .reset_index(name='평균입차대수')
            .rename(columns={'입차_시': 'hour'})
        )

        return avg_per_hour