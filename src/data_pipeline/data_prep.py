import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Iterable
from .result_checker import ResultChecker
from ._helpers import DataHandler

class PreProcessing:
    # 가용면 계산 시 제외할 전용면 후보 컬럼
    DEFAULT_RESERVED_COLS = ['장애인', '친환경면수 합계']
    
    def __init__(
            self, 
            dongnae: pd.DataFrame,
            parking_info: pd.DataFrame, 
            lot_name: str,
            exit_first_when_same_time: bool,
            ensure_all_hours: bool,
            reserved_cols: Optional[Iterable[str]] = None    
        ):
        self.lot_name = lot_name
        self.dongnae = dongnae.copy()
        self.parking_info = parking_info.copy()
        self.exit_first_when_same_time = exit_first_when_same_time
        self.ensure_all_hours = ensure_all_hours
        self.reserved_cols = reserved_cols
        self.last_result: Optional[Dict[str, pd.DataFrame]] = None
        
    def convert_datetime_cols(self) -> "PreProcessing":
        """
        원본 사용 데이터(raw_usage)의 '입차시간', '출차시간' 컬럼을 datetime 형식으로 변환합니다.

        동작:
            - 컬럼이 존재하면 dtype을 검사합니다.
            - datetime 형식이 아닐 경우 pd.to_datetime으로 안전 변환(errors='coerce')을 수행합니다.
            - 변환 실패 값은 NaT으로 처리됩니다.

        Returns:
             PreProcessing: datetime 변환 결과를 포함한 자기 자신 인스턴스
        """
        # for col in ('입차시간', '출차시간'):
        #     if col in self.dongnae.columns:
        #         if not np.issubdtype(self.dongnae[col].dtype, np.datetime64):
        #             self.dongnae[col] = pd.to_datetime(self.dongnae[col], errors='coerce')
        self.dongnae['입차시간'] = pd.to_datetime(self.dongnae['입차일시'],  format="mixed", errors="raise")
        self.dongnae['출차시간'] = pd.to_datetime(self.dongnae['출차일시'],  format="mixed", errors="raise")

        return self

    def filter_by_lot(self) -> "PreProcessing":
        """
        주차장명을 기준으로 dongnae 데이터에서 특정 주차장의 행만 선택합니다.
        
        Returns:
            PreProcessing: 선택된 결과를 포함한 자기 자신 인스턴스
        """
        self.sel_dongnae: pd.DataFrame = self.dongnae[self.dongnae['주차장명'] == self.lot_name]

        return self

    def combine_events(self) -> "PreProcessing":
        """
        선택된 데이터(sel_dongnae)에서 입차/출차 기록을 각각 정리하여 하나의 DataFrame으로 결합합니다.

        - 입차: '입차시간'을 '기준시간'으로 컬럼명 변경하고, '입출구분'을 '입차'로 표시
        - 출차: '출차시간'을 '기준시간'으로 컬럼명 변경하고, '입출구분'을 '출차'로 표시

        Returns:
            PreProcessing: 결합된 결과 DataFrame(combined_df)을 속성으로 포함한 자기 자신 인스턴스
        """

        self.entries: pd.DataFrame = (
            self.sel_dongnae.loc[self.sel_dongnae['입차시간'].notna(), ['차량번호', '입차유형', '입차시간']]
            .rename(columns={'입차시간': '기준시간'})
            .assign(입출구분='입차')
            )
        
        self.exits: pd.DataFrame = (
            self.sel_dongnae.loc[self.sel_dongnae['출차시간'].notna(), ['차량번호', '입차유형', '출차시간']] 
            .rename(columns={'출차시간': '기준시간'})
            .assign(입출구분='출차')
            )
            

        self.combined_df: pd.DataFrame = pd.concat([self.entries, self.exits], ignore_index=True) 

        return self 

    def sort_events(self) -> "PreProcessing":
        """
        이벤트 데이터(combined_df)를 기준시간으로 정렬합니다.
        동일한 시간대에서는 설정(exit_first)에 따라 출차를 먼저 처리할 수 있습니다.
        """

        if self.exit_first_when_same_time:
            self.combined_df['sort_key'] = (self.combined_df['입출구분']=='입차').astype(int)
            self.combined_df = self.combined_df.sort_values(['기준시간', 'sort_key'], kind='mergesort')
            self.combined_df = self.combined_df.drop(columns='sort_key')
        else:
            self.combined_df = self.combined_df.sort_values('기준시간', kind='mergesort')

        return self   
    
    def calculate_parking_status(self) -> "PreProcessing": 
        """
        입출차 이벤트 기반으로 주차 점유 상태를 계산하여 컬럼을 추가합니다.

        수행 내용:
            - '차량변화'(입차=+1, 출차=-1), '누적차량수' 계산
            - '날짜', '시간대' 파생 컬럼 생성
            - 가용면 수(available_slots) 산출 후 '만차상태' 여부 계산
            - 상태 변화 기준으로 '만차그룹' 구간 번호 부여

        Returns:
            PreProcessing: 계산 결과가 반영된 자기 자신 인스턴스
        """
         
        # 누적 계산
        self.combined_df['차량변화']   = self.combined_df['입출구분'].map({'입차': 1, '출차': -1})
        self.combined_df['누적차량수'] = self.combined_df['차량변화'].cumsum()
        self.combined_df['날짜']      = self.combined_df['기준시간'].dt.date
        self.combined_df['시간대']    = self.combined_df['기준시간'].dt.hour

        # 가용면 계산
        available_slots = self._compute_available_slots()

        # 만차 상태/그룹
        self.combined_df['만차상태']  = self.combined_df['누적차량수'] >= available_slots
        self.combined_df['만차그룹']  = self.combined_df['만차상태'].ne(self.combined_df['만차상태'].shift()).cumsum()

        return self

    def compute_full_periods(self) -> "PreProcessing":
        """
        만차 상태(True)인 구간의 시작~종료 시각을 계산합니다.

        동작:
            - '만차그룹' 기준으로 구간 시작('시작')과 상태('만차상태')를 집계
            - 다음 그룹 시작 시각을 '종료'로 사용 (마지막 그룹은 데이터 최종 시각으로 보정)
            - 종료 시각이 시작 시각보다 큰 구간만 필터링

        Args:
            combined_df (pd.DataFrame): 만차그룹, 기준시간, 만차상태 컬럼을 포함한 이벤트 데이터

        Returns:
            pd.DataFrame: 만차 구간의 시작/종료 시각을 담은 DataFrame
                - 컬럼: ['시작', '종료']
        """
        groups: pd.DataFrame = (
                self.combined_df.groupby('만차그룹')
                .agg(시작=('기준시간', 'first'), 상태=('만차상태', 'first'))
                .reset_index(drop=True)
                )

        groups['다음시작'] = groups['시작'].shift(-1)
        analysis_end: datetime = self.combined_df['기준시간'].max()

        self.full_periods: pd.DataFrame = groups[groups['상태']].copy()
        self.full_periods['종료'] = self.full_periods['다음시작'].fillna(analysis_end)
        self.full_periods: pd.DataFrame = self.full_periods.loc[
            self.full_periods['종료'] > self.full_periods['시작'], ['시작', '종료']
        ].reset_index(drop=True)

        return self

    def build_hourly_and_summaries(self) -> "PreProcessing":
        """
        만차 구간(full_periods)을 시간(시) 경계로 분할하여 hourly_detail을 생성하고,
        전체/평일/주말 집계와 일수 테이블을 계산해 인스턴스 속성으로 저장합니다.

        동작:
            - 만차 구간 [시작, 종료) 를 시(hour) 경계로 잘라 행 단위로 확장(hourly_detail)
            - hourly_detail과 combined_df를 이용해
            · 전체 집계: hourly_summary, daily_summary
            · 평일/주말 집계: hourly_summary_weekday/weekend, daily_summary_weekday/weekend
            - 분석 기간 내 일수 집계: day_counts(전체/평일/주말), day_counts_by_dow(요일별)

        Args:
            full_periods (pd.DataFrame): 만차 구간의 시작/종료 시각을 담은 DataFrame
                - 필수 컬럼: ['시작', '종료'] (datetime)
            combined_df (pd.DataFrame): 이벤트 기반의 기준 데이터
                - 예: ['기준시간', '만차상태', ...]
            ensure_all_hours (bool, optional): 모든 시간대(0~23) 라인을 집계 결과에 보장할지 여부. 기본 True.

        Returns:
            PreProcessing: 계산된 결과가 속성에 반영된 자기 자신 인스턴스
                - self.hourly_detail
                - self.hourly_summary, self.daily_summary
                - self.hourly_summary_weekday, self.hourly_summary_weekend
                - self.daily_summary_weekday, self.daily_summary_weekend
                - self.day_counts, self.day_counts_by_dow
        """
        # 만차 구간을 시간(시) 경계로 분할 → hourly_detail
        hourly_rows = []
        for s, e in zip(self.full_periods['시작'], self.full_periods['종료']):
            current = s
            while current < e:
                period_end = min(PreProcessing.next_hour_mark(current), e)
                minutes = (period_end - current).total_seconds() / 60.0
                hourly_rows.append({
                    '날짜': current.date(),
                    '시간대': current.hour,
                    '유지시간_분': minutes
                })
                current = period_end

        self.hourly_detail: pd.DataFrame = pd.DataFrame(
            hourly_rows, columns=['날짜', '시간대', '유지시간_분']
        )

        # 요일 추가 
        self.hourly_detail = DataHandler.add_weekday_column(self.hourly_detail)
        # 집계(전체)
        self._compute_summaries()
        # 집계(평일/주말 분리)
        self._aggregate_by_week_segment()
        # 일수 테이블(전체/평일/주말 + 요일별)
        self._compute_day_counts()

        return self

    def package_results(self) -> Dict[str, pd.DataFrame]:
        """
        분석 산출물(DataFrame)들을 표준 키로 묶어 딕셔너리로 패키징하고,
        self.last_result에 저장한 뒤 반환합니다.

        - 설명:
            combined_df (pd.DataFrame): 이벤트 결합 데이터 (index는 초기화하여 저장)
            full_periods (pd.DataFrame): 만차 구간 시작/종료
            hourly_detail (pd.DataFrame): 시간(시) 경계로 확장한 상세 구간
            hourly_summary (pd.DataFrame): 전체 시간대 요약
            daily_summary (pd.DataFrame): 전체 일자 요약
            hourly_summary_weekday (pd.DataFrame): 평일 시간대 요약
            hourly_summary_weekend (pd.DataFrame): 주말 시간대 요약
            daily_summary_weekday (pd.DataFrame): 평일 일자 요약
            daily_summary_weekend (pd.DataFrame): 주말 일자 요약
            day_counts (pd.DataFrame): 전체/평일/주말 일수 집계
            day_counts_by_dow (pd.DataFrame): 요일별 일수 집계

        Returns:
            Dict[str, pd.DataFrame]: 표준 키로 정리된 결과 딕셔너리
                - 'combined_df', 'full_periods', 'hourly_detail',
                'hourly_summary', 'daily_summary',
                'hourly_summary_weekday', 'hourly_summary_weekend',
                'daily_summary_weekday', 'daily_summary_weekend',
                'day_counts', 'day_counts_by_dow'
        """
        self.last_result: Dict[str, pd.DataFrame] = {
            'combined_df': self.combined_df.reset_index(drop=True),
            'full_periods': self.full_periods,
            'hourly_detail': self.hourly_detail,
            'hourly_summary': self.hourly_summary,
            'daily_summary': self.daily_summary,
            'hourly_summary_weekday': self.hourly_summary_weekday,
            'hourly_summary_weekend': self.hourly_summary_weekend,
            'daily_summary_weekday': self.daily_summary_weekday,
            'daily_summary_weekend': self.daily_summary_weekend,
            'day_counts': self.day_counts,
            'day_counts_by_dow': self.day_counts_by_dow,
        }

        return self
    
    def run_pipeline(self):
         # === Public pipeline steps (체이닝에서 직접 호출) ========================
        result = (
                self.convert_datetime_cols()
                .filter_by_lot()
                .combine_events()
            )
        
        # 조건 처리
        last_result = ResultChecker.empty_if_no_events(result.combined_df)
        
        result = (result.sort_events()
         .calculate_parking_status()
         .compute_full_periods()
         .build_hourly_and_summaries()
         .package_results()
        )
        
        return result 
    
    
    # === Internal helpers (런 파이프라인 내부 전용) ==========================
    @staticmethod
    def next_hour_mark(ts: pd.Timestamp) -> pd.Timestamp:
        """항상 '다음 정시(시:00)' 시각. 정시인 경우 +1시간."""
        return ts.floor('H') + pd.Timedelta(hours=1)

    @staticmethod
    def safe_div(a: pd.Series, b) -> pd.Series:
        """0 나눗셈/무한대 안전 처리"""
        return a.divide(b).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    @staticmethod
    def is_weekend(ts_like: pd.Series) -> pd.Series:
        """주말(True) 여부 (토=5, 일=6)"""
        wd = pd.to_datetime(ts_like).dt.weekday  # Mon=0 ... Sun=6
        return wd >= 5
    
    def _compute_available_slots(self) -> int:
        """
        주차장 정보(parking_info)에서 현재 가용면(= 현재 주차면 - 전용면 합)을 계산합니다.

        동작:
            - self.lot_name에 해당하는 행을 조회하고 숫자 컬럼을 안전하게 변환합니다.
            - 전용면 컬럼 목록(self.reserved_cols)이 없으면 클래스 기본값(DEFAULT_RESERVED_COLS)을 사용합니다.
            - 존재하지 않는 전용면 컬럼은 자동으로 제외합니다.
            - (방어) 계산 결과가 음수가 되면 0으로 보정합니다.
            - (부수효과) self.reserved_cols를 정규화하여 인스턴스에 다시 저장합니다.

        Returns:
            int: 가용면 수(0 이상의 정수)

        Raises:
            ValueError: parking_info에 self.lot_name에 해당하는 데이터가 없을 때
        """
        lot_row: pd.DataFrame = self.parking_info[self.parking_info['주차장명'] == self.lot_name].copy()
        print(f"주차장 추출: {lot_row['주차장명'].iloc[0]}")

        if lot_row.empty:
            raise ValueError(f"parking_info에 '{self.lot_name}' 데이터가 없습니다.")
 

        total_slots: int = pd.to_numeric(lot_row['현재 주차면'], errors='coerce').iloc[0]
        self.reserved_cols: List = list(self.reserved_cols) if self.reserved_cols is not None else self.__class__.DEFAULT_RESERVED_COLS
        self.reserved_cols: List = [c for c in self.reserved_cols if c in lot_row.columns]

        if self.reserved_cols:
            lot_row[self.reserved_cols]= lot_row[self.reserved_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            reserved_sum: int = lot_row[self.reserved_cols].sum(axis=1).iloc[0]
        else:
            reserved_sum = 0

        available: int = int(total_slots - reserved_sum)
        print(f"가용면 수: {max(0, available)}")

        return max(0, available)  # 방어
    
    def _compute_summaries(self) -> None:
        """
        만차 구간 데이터를 바탕으로 시간대별/일자별 요약 통계를 계산합니다.

        동작:
            - 시간대별 총 유지시간, 만차발생일수, 발생횟수, 만차시작횟수 집계
            - 발생일 기준/전체일 기준 평균값 계산
            - 0~23시 모든 시간대를 강제로 포함 (ensure_all_hours 옵션)
            - 날짜별 총 유지시간 집계

        Returns:
            PreProcessing: 집계 결과를 속성으로 업데이트한 자기 자신 인스턴스
                - self.hourly_summary
                - self.daily_summary
        """
        total_days: List = self.combined_df['날짜'].nunique()

        if self.hourly_detail.empty:
            self.hourly_summary, self.daily_summary = ResultChecker._build_hourly_result(self.hourly_detail)
            return self
        
        else:
            # 시간대별 총 유지시간 & 만차발생일수
            hourly_agg = (self.hourly_detail.groupby('시간대')
                        .agg(총유지시간_분=('유지시간_분', 'sum'),
                            만차발생일수=('날짜', 'nunique'))
                        .reset_index())
            
            # 요일별 총 유지시간 & 만차발생일수
            self.dow_agg = (self.hourly_detail.groupby('요일')
                        .agg(총유지시간_분=('유지시간_분', 'sum'),
                            만차발생일수=('날짜', 'nunique'))
                        .reset_index())

            # 시간대별 '만차 발생 횟수'(조각 수)
            occ_total = (self.hourly_detail.groupby('시간대')
                        .size()
                        .rename('발생횟수_총').reset_index())
            
            # 요일별 '만차 발생 횟수'(조각 수)
            self.dow_total = (self.hourly_detail.groupby('요일')
                        .size()
                        .rename('발생횟수_총').reset_index())


            # 시간대별 '만차 시작 횟수'(False -> True 전이)
            starts_mask = self.combined_df['만차상태'] & (~self.combined_df['만차상태'].shift(fill_value=False))
            start_hours = self.combined_df.loc[starts_mask, '기준시간'].dt.hour
            start_counts = (start_hours.value_counts()
                            .rename_axis('시간대').reset_index(name='만차시작횟수_총'))

            self.hourly_summary = (hourly_agg
                            .merge(occ_total, on='시간대', how='outer')
                            .merge(start_counts, on='시간대', how='outer')
                            .fillna({'총유지시간_분': 0.0, '만차발생일수': 0,
                                    '발생횟수_총': 0, '만차시작횟수_총': 0}))
            
            # 요일-평균(평균유지시간_분(발생일기준))
            self.dow_agg['평균유지시간_분(발생일기준)'] = self.safe_div(
                self.dow_agg['총유지시간_분'], self.dow_agg['만차발생일수']
                )

            # 평균(발생일/전체일)
            self.hourly_summary['평균유지시간_분(발생일기준)'] = self.safe_div(
                self.hourly_summary['총유지시간_분'], self.hourly_summary['만차발생일수']
            )
            self.hourly_summary['평균유지시간_분(전체일기준)'] = self.safe_div(
                self.hourly_summary['총유지시간_분'], pd.Series(total_days, index=self.hourly_summary.index)
            )
            self.hourly_summary['평균발생횟수(발생일기준)'] = self.safe_div(
                self.hourly_summary['발생횟수_총'], self.hourly_summary['만차발생일수']
            )
            self.hourly_summary['평균발생횟수(전체일기준)'] = self.safe_div(
                self.hourly_summary['발생횟수_총'], pd.Series(total_days, index=self.hourly_summary.index)
            )
            self.hourly_summary['평균만차시작횟수(발생일기준)'] = self.safe_div(
                self.hourly_summary['만차시작횟수_총'], self.hourly_summary['만차발생일수']
            )
            self.hourly_summary['평균만차시작횟수(전체일기준)'] = self.safe_div(
                self.hourly_summary['만차시작횟수_총'], pd.Series(total_days, index=self.hourly_summary.index)
            )

            # 0~23시 강제 포함
            if self.ensure_all_hours:
                all_hours = pd.DataFrame({'시간대': range(24)})
                self.hourly_summary = all_hours.merge(self.hourly_summary, on='시간대', how='left').fillna(0)

            self.hourly_summary = self.hourly_summary.sort_values('시간대').reset_index(drop=True)

            # 날짜별 총 유지시간
            self.daily_summary = (self.hourly_detail.groupby('날짜')['유지시간_분']
                            .sum().reset_index())
        
    def _aggregate_by_week_segment(self) -> None:
        """
        평일/주말로 구분된 시간대 요약 및 날짜 요약 통계를 계산합니다.

        동작:
            - hourly_detail 데이터를 바탕으로 평일/주말 구분 열을 생성
            - 시간대별 총 유지시간, 만차발생일수, 발생횟수, 만차시작횟수 집계
            - 발생일 기준/전체일 기준 평균값 계산
            - ensure_all_hours=True일 경우, 평일/주말 × 24시간대 모두 포함
            - 일자별 총 유지시간 집계 (평일/주말 분리)

        Returns:
             해당 내용 속성을 업데이트한 자기 자신 인스턴스
                - pd.DataFrame: hourly_summary_weekday (평일 시간대 요약)
                - pd.DataFrame: hourly_summary_weekend (주말 시간대 요약)
                - pd.DataFrame: daily_summary_weekday (평일 일자 요약)
                - pd.DataFrame: daily_summary_weekend (주말 일자 요약)
        """
        
        # 시간대별 총유지/발생일수
        hd = self.hourly_detail.copy()
        hd_dt = pd.to_datetime(hd['날짜'])  # date -> datetime
        hd['구분'] = np.where(PreProcessing.is_weekend(hd_dt), '주말', '평일')

        hourly_agg = (hd.groupby(['구분', '시간대'])
                        .agg(총유지시간_분=('유지시간_분', 'sum'),
                             만차발생일수=('날짜', 'nunique'))
                        .reset_index())

        # 시간대별 '만차 발생 횟수'(조각 수)
        occ_total = (hd.groupby(['구분','시간대'])
                       .size()
                       .rename('발생횟수_총')
                       .reset_index())

        # 시간대별 '만차 시작 횟수'
        starts_mask = self.combined_df['만차상태'] & (~self.combined_df['만차상태'].shift(fill_value=False))
        starts = self.combined_df.loc[starts_mask, ['기준시간']].copy()
        starts['구분'] = np.where(PreProcessing.is_weekend(starts['기준시간']), '주말', '평일')
        starts['시간대'] = starts['기준시간'].dt.hour
        start_counts = (starts.groupby(['구분','시간대'])
                              .size()
                              .rename('만차시작횟수_총')
                              .reset_index())

        seg_hourly = (hourly_agg
                      .merge(occ_total,   on=['구분','시간대'], how='outer')
                      .merge(start_counts, on=['구분','시간대'], how='outer')
                      .fillna({'총유지시간_분':0.0, '만차발생일수':0,
                               '발생횟수_총':0, '만차시작횟수_총':0}))

        # 세그먼트별 총일수(그 구분에서 관측된 서로 다른 날짜 수)
        dates = self.combined_df[['기준시간']].copy()
        dates['날짜'] = dates['기준시간'].dt.date
        dates['구분'] = np.where(PreProcessing.is_weekend(dates['기준시간']), '주말', '평일')
        days_by_group = (dates.drop_duplicates(subset=['구분','날짜'])
                              .groupby('구분')['날짜']
                              .nunique())

        seg_hourly['총일수_그룹'] = seg_hourly['구분'].map(days_by_group).fillna(0).astype(int)

        # 평균 컬럼들
        seg_hourly['평균유지시간_분(발생일기준)'] = self.safe_div(
            seg_hourly['총유지시간_분'], seg_hourly['만차발생일수']
        )
        seg_hourly['평균유지시간_분(전체일기준)'] = self.safe_div(
            seg_hourly['총유지시간_분'], seg_hourly['총일수_그룹']
        )
        seg_hourly['평균발생횟수(발생일기준)'] = self.safe_div(
            seg_hourly['발생횟수_총'], seg_hourly['만차발생일수']
        )
        seg_hourly['평균발생횟수(전체일기준)'] = self.safe_div(
            seg_hourly['발생횟수_총'], seg_hourly['총일수_그룹']
        )
        seg_hourly['평균만차시작횟수(발생일기준)'] = self.safe_div(
            seg_hourly['만차시작횟수_총'], seg_hourly['만차발생일수']
        )
        seg_hourly['평균만차시작횟수(전체일기준)'] = self.safe_div(
            seg_hourly['만차시작횟수_총'], seg_hourly['총일수_그룹']
        )

        # 추가
        seg_daily = ResultChecker.ensure_all_time_slots(seg_hourly, days_by_group, self.ensure_all_hours)

        # 분리(평일/주말) 테이블
        self.hourly_summary_weekday = seg_hourly.loc[seg_hourly['구분']=='평일'].drop(columns=['구분','총일수_그룹']).reset_index(drop=True)
        self.hourly_summary_weekend = seg_hourly.loc[seg_hourly['구분']=='주말'].drop(columns=['구분','총일수_그룹']).reset_index(drop=True)

        # 날짜별 총 유지시간 (평일/주말 분리)
        seg_daily = (hd.groupby(['구분','날짜'])['유지시간_분']
                       .sum()
                       .reset_index()
                       .sort_values(['구분','날짜'])
                       .reset_index(drop=True))

        self.daily_summary_weekday = seg_daily.loc[seg_daily['구분']=='평일', ['날짜','유지시간_분']].reset_index(drop=True)
        self.daily_summary_weekend = seg_daily.loc[seg_daily['구분']=='주말', ['날짜','유지시간_분']].reset_index(drop=True)

    def _compute_day_counts(self) -> None:
        """
        날짜(일) 단위 고유 일수 계산:
          - day_counts: 전체/평일/주말 일수
          - day_counts_by_dow: 요일별(월~일) 일수
        """
        # 고유 날짜(정규화: 00:00로)
        days = self.combined_df['기준시간'].dt.normalize().drop_duplicates()
        total_days = int(len(days))

        # 평일/주말 분리 (Mon=0 ... Sun=6)
        wd = days.dt.weekday
        weekday_days = int((wd < 5).sum())
        weekend_days = int((wd >= 5).sum())

        self.day_counts = pd.DataFrame({
            '구분': ['전체', '평일', '주말'],
            '일수': [total_days, weekday_days, weekend_days]
        })

        # 요일별 일수
        labels = np.array(['월','화','수','목','금','토','일'])
        self.dow_counts = (wd.value_counts()
                        .reindex(range(7), fill_value=0)
                        .sort_index()
                        .astype(int))
        self.day_counts_by_dow = pd.DataFrame({
            '요일': labels,
            '일수': self.dow_counts.values
        })