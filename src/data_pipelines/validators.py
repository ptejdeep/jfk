"""
Data Quality Validators for JFKIAT Operations
Ensures data integrity before forecasting and reporting.
"""

import pandas as pd
import logging
from typing import Tuple, List, Dict

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validates data quality and integrity for forecasting pipelines.
    """

    @staticmethod
    def validate_forecast_input(df: pd.DataFrame, datetime_col: str, metric_col: str) -> Tuple[bool, List[str]]:
        """
        Validate input data for forecasting.
        
        Args:
            df: Input DataFrame
            datetime_col: Name of datetime column
            metric_col: Name of metric column
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check columns exist
        if datetime_col not in df.columns:
            issues.append(f"Missing datetime column: {datetime_col}")
        if metric_col not in df.columns:
            issues.append(f"Missing metric column: {metric_col}")
        
        if issues:
            return False, issues
        
        # Check minimum data points (need at least 30 days of hourly data)
        if len(df) < 720:
            issues.append(f"Insufficient data: {len(df)} rows (minimum 720 required for 30 days hourly)")
        
        # Check for null values
        null_datetime = df[datetime_col].isnull().sum()
        null_metric = df[metric_col].isnull().sum()
        
        if null_datetime > len(df) * 0.01:  # Allow < 1% nulls
            issues.append(f"Too many null values in datetime: {null_datetime} ({null_datetime/len(df)*100:.2f}%)")
        
        if null_metric > len(df) * 0.05:  # Allow < 5% nulls
            issues.append(f"Too many null values in metric: {null_metric} ({null_metric/len(df)*100:.2f}%)")
        
        # Check datetime column is actually datetime-like
        try:
            pd.to_datetime(df[datetime_col])
        except Exception as e:
            issues.append(f"Cannot parse datetime column: {str(e)}")
        
        # Check metric is numeric
        try:
            pd.to_numeric(df[metric_col], errors='coerce')
            percent_numeric = pd.to_numeric(df[metric_col], errors='coerce').notna().sum() / len(df) * 100
            if percent_numeric < 95:
                issues.append(f"Metric column has only {percent_numeric:.1f}% numeric values")
        except Exception as e:
            issues.append(f"Cannot parse metric column as numeric: {str(e)}")
        
        # Check for reasonable metric ranges
        metric_numeric = pd.to_numeric(df[metric_col], errors='coerce')
        if (metric_numeric < 0).any():
            issues.append("Metric contains negative values (not expected for passenger/vehicle counts)")
        
        if (metric_numeric > 100000).any():
            logger.warning("Metric contains very large values (> 100,000) - verify data source")
        
        is_valid = len(issues) == 0
        return is_valid, issues

    @staticmethod
    def validate_forecast_output(forecast_df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate forecast output meets quality standards.
        
        Args:
            forecast_df: Output DataFrame from forecaster
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        required_cols = ['timestamp', 'forecast', 'lower_bound', 'upper_bound']
        missing_cols = [col for col in required_cols if col not in forecast_df.columns]
        
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
            return False, issues
        
        # Check no null forecasts
        if forecast_df['forecast'].isnull().any():
            issues.append(f"Forecast contains {forecast_df['forecast'].isnull().sum()} null values")
        
        # Check bounds are valid
        if (forecast_df['lower_bound'] > forecast_df['forecast']).any():
            issues.append("Lower bound exceeds forecast value in some rows")
        
        if (forecast_df['upper_bound'] < forecast_df['forecast']).any():
            issues.append("Upper bound is less than forecast value in some rows")
        
        # Check reasonable confidence interval width
        forecast_df['interval_width'] = forecast_df['upper_bound'] - forecast_df['lower_bound']
        if (forecast_df['interval_width'] <= 0).any():
            issues.append("Confidence interval width is zero or negative in some rows")
        
        is_valid = len(issues) == 0
        return is_valid, issues

    @staticmethod
    def check_data_freshness(df: pd.DataFrame, datetime_col: str, max_hours_stale: int = 24) -> Tuple[bool, str]:
        """
        Check if data is fresh enough for forecasting.
        
        Args:
            df: DataFrame to check
            datetime_col: Name of datetime column
            max_hours_stale: Maximum acceptable staleness in hours
            
        Returns:
            Tuple of (is_fresh, message)
        """
        from datetime import datetime, timedelta
        
        latest_timestamp = pd.to_datetime(df[datetime_col]).max()
        hours_old = (datetime.now(latest_timestamp.tz) - latest_timestamp).total_seconds() / 3600
        
        is_fresh = hours_old <= max_hours_stale
        message = f"Data is {hours_old:.1f} hours old (threshold: {max_hours_stale} hours)"
        
        return is_fresh, message

    @staticmethod
    def check_outliers(df: pd.DataFrame, metric_col: str, z_threshold: float = 3.0) -> Dict[str, any]:
        """
        Detect statistical outliers using z-score method.
        
        Args:
            df: DataFrame to check
            metric_col: Name of metric column
            z_threshold: Z-score threshold (default 3.0 = 0.3% outliers)
            
        Returns:
            Dictionary with outlier statistics
        """
        from scipy import stats
        
        metric_values = pd.to_numeric(df[metric_col], errors='coerce').dropna()
        z_scores = abs(stats.zscore(metric_values))
        outliers = (z_scores > z_threshold).sum()
        
        return {
            'outlier_count': int(outliers),
            'outlier_percent': float(outliers / len(metric_values) * 100),
            'threshold': z_threshold,
            'mean': float(metric_values.mean()),
            'std': float(metric_values.std()),
            'min': float(metric_values.min()),
            'max': float(metric_values.max())
        }


class DataQualityReport:
    """
    Generate comprehensive data quality reports.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.report = {}

    def generate_report(self, datetime_col: str, metric_col: str) -> Dict:
        """
        Generate complete data quality report.
        
        Args:
            datetime_col: Name of datetime column
            metric_col: Name of metric column
            
        Returns:
            Dictionary with comprehensive report
        """
        validator = DataValidator()
        
        # Basic validation
        is_valid, issues = validator.validate_forecast_input(self.df, datetime_col, metric_col)
        
        # Freshness check
        is_fresh, freshness_msg = validator.check_data_freshness(self.df, datetime_col)
        
        # Outlier analysis
        outlier_stats = validator.check_outliers(self.df, metric_col)
        
        self.report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'total_rows': len(self.df),
            'validation': {
                'is_valid': is_valid,
                'issues': issues
            },
            'freshness': {
                'is_fresh': is_fresh,
                'message': freshness_msg
            },
            'outliers': outlier_stats,
            'columns': {
                'datetime_col': datetime_col,
                'metric_col': metric_col,
                'total_columns': len(self.df.columns)
            }
        }
        
        return self.report

    def print_report(self):
        """Print formatted data quality report to console."""
        print("\n" + "="*60)
        print("DATA QUALITY REPORT")
        print("="*60)
        print(f"Timestamp: {self.report['timestamp']}")
        print(f"Total Rows: {self.report['total_rows']}")
        print()
        
        print("VALIDATION:")
        print(f"  Valid: {self.report['validation']['is_valid']}")
        if self.report['validation']['issues']:
            for issue in self.report['validation']['issues']:
                print(f"  ⚠️  {issue}")
        else:
            print("  ✓ No issues found")
        print()
        
        print("FRESHNESS:")
        print(f"  Fresh: {self.report['freshness']['is_fresh']}")
        print(f"  {self.report['freshness']['message']}")
        print()
        
        print("OUTLIERS (Z-score > 3.0):")
        print(f"  Count: {self.report['outliers']['outlier_count']}")
        print(f"  Percent: {self.report['outliers']['outlier_percent']:.2f}%")
        print()
        
        print("STATISTICS:")
        print(f"  Mean: {self.report['outliers']['mean']:.2f}")
        print(f"  Std Dev: {self.report['outliers']['std']:.2f}")
        print(f"  Min: {self.report['outliers']['min']:.2f}")
        print(f"  Max: {self.report['outliers']['max']:.2f}")
        print("="*60 + "\n")

    def export_report_json(self, filepath: str):
        """Export report to JSON file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.report, f, indent=2, default=str)
        logger.info(f"Report exported to {filepath}")
