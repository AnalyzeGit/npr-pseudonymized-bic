
import pandas as pd
from typing import Dict


class ResultChecker:
    def __init__(self):
        ...

    @staticmethod
    def empty_if_no_events(combined_df: pd.DataFrame):
        if combined_df.empty:
            return ResultChecker._build_empty_result()
        
    @staticmethod
    def init_empty_hourly_result(hourly_detail: pd.DataFrame):
        """
        hourly_detail이 비어 있는 경우, 시간대별/일자별 요약 결과를 기본 빈 구조로 초기화합니다.

        동작:
            - 시간대별(hourly_summary): 0~23시를 모두 포함한 0 값 테이블 생성
            - 일자별(daily_summary): 컬럼만 존재하는 빈 DataFrame 생성
            - 자기 자신 인스턴스를 반환하여 메서드 체이닝 지원

        Returns:
            PreProcessing: 빈 요약 결과가 설정된 자기 자신 인스턴스
                - self.hourly_summary
                - self.daily_summary
        """
        return ResultChecker._build_hourly_result()


    @staticmethod
    def ensure_all_time_slots(
            seg_hourly: pd.DataFrame, 
            days_by_group: pd.DataFrame, 
            ensure_all_hours: bool
            ) -> pd.DataFrame :
        """
        평일/주말 × 24시간대 조합이 모두 포함되도록 seg_hourly를 확장하고,
        누락된 시간대는 0으로 채웁니다.

        Args:
            seg_hourly (pd.DataFrame): 구분(평일/주말), 시간대별 집계가 담긴 DataFrame
            days_by_group (pd.DataFrame or pd.Series): 구분별 총일수 정보
            ensure_all_hours (bool): True일 경우, 0~23시 전체 시간대를 강제 포함

        Returns:
            pd.DataFrame: 구분 × 시간대(0~23) 전체가 포함된 정렬된 DataFrame
        """
        if ensure_all_hours:
            idx = pd.MultiIndex.from_product([['평일','주말'], range(24)], names=['구분','시간대'])
            seg_hourly = (seg_hourly
                          .set_index(['구분','시간대'])
                          .reindex(idx, fill_value=0)
                          .reset_index())
            seg_hourly['총일수_그룹'] = seg_hourly['구분'].map(days_by_group).fillna(0).astype(int)

        return seg_hourly.sort_values(['구분','시간대']).reset_index(drop=True)


    def _build_empty_result() -> Dict[str, pd.DataFrame]:
        empty_hourly = pd.DataFrame({
            '시간대': range(24),
            '총유지시간_분': 0.0,
            '만차발생일수': 0,
            '평균유지시간_분(발생일기준)': 0.0,
            '평균유지시간_분(전체일기준)': 0.0,
            '발생횟수_총': 0,
            '평균발생횟수(발생일기준)': 0.0,
            '평균발생횟수(전체일기준)': 0.0,
            '만차시작횟수_총': 0,
            '평균만차시작횟수(발생일기준)': 0.0,
            '평균만차시작횟수(전체일기준)': 0.0
        })
        return {
            'combined_df': pd.DataFrame(),
            'full_periods': pd.DataFrame(columns=['시작', '종료']),
            'hourly_detail': pd.DataFrame(columns=['날짜', '시간대', '유지시간_분']),
            'hourly_summary': empty_hourly.copy(),
            'daily_summary': pd.DataFrame(columns=['날짜', '유지시간_분']),
            'hourly_summary_weekday': empty_hourly.copy(),
            'hourly_summary_weekend': empty_hourly.copy(),
            'daily_summary_weekday': pd.DataFrame(columns=['날짜', '유지시간_분']),
            'daily_summary_weekend': pd.DataFrame(columns=['날짜', '유지시간_분']),
            'day_counts': pd.DataFrame({'구분': ['전체','평일','주말'], '일수': [0,0,0]}),
            'day_counts_by_dow': pd.DataFrame({'요일': ['월','화','수','목','금','토','일'], '일수': [0]*7})
        }
    
    def _build_hourly_result() -> Dict[str, pd.DataFrame]:
        hourly_summary = pd.DataFrame({
                    '시간대': range(24),
                    '총유지시간_분': 0.0,
                    '만차발생일수': 0,
                    '평균유지시간_분(발생일기준)': 0.0,
                    '평균유지시간_분(전체일기준)': 0.0,
                    '발생횟수_총': 0,
                    '평균발생횟수(발생일기준)': 0.0,
                    '평균발생횟수(전체일기준)': 0.0,
                    '만차시작횟수_총': 0,
                    '평균만차시작횟수(발생일기준)': 0.0,
                    '평균만차시작횟수(전체일기준)': 0.0
                })
        daily_summary = pd.DataFrame(columns=['날짜', '유지시간_분'])

        return hourly_summary, daily_summary