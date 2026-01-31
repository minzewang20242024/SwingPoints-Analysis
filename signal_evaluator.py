import pandas as pd
import numpy as np
from typing import Optional

class SignalEvaluator:
    """
    Signal Evaluation Class (Optimized for Large-Cap Tech Stocks)
    Functionality: Verify true/false breakout/breakdown signals, calculate performance metrics, export results.
    Core data is initialized via __init__, and users can customize verification parameters.
    """

    def __init__(self, holding_period: int = 10, min_threshold: float = 0.03, vol_multiplier: float = 1.5):
        """
        Class Initialization: Initialize core verification parameters and result storage.
        :param holding_period: Signal verification holding period (default 10, optimized value for large-cap tech stocks after justification)
        :param min_threshold: Minimum price change threshold (default 0.03=3%, filter minor fluctuations)
        :param vol_multiplier: Volume synergy multiplier (default 1.5, verify capital consensus)
        """
        # Core parameter initialization (default values justified for large-cap tech stocks)
        self.holding_period = holding_period
        self.min_threshold = min_threshold
        self.vol_multiplier = vol_multiplier

        # Core data initialization (passed via subsequent methods for flexibility)
        self.prices = pd.Series()
        self.volume = pd.Series()
        self.uptrend_line = None
        self.downtrend_line = None

        # Evaluation result storage
        self.evaluation_result = {}
        self.breakout_verification = []
        self.breakdown_verification = []

    def _verify_breakout_single(self, signal_idx: int) -> str:
        """
        Helper Method: Verify true/false of a single candidate breakout signal (holding period + price change threshold).
        :param signal_idx: Signal index
        :return: Verification result ("TPB" = True Positive Breakout, "FPB" = False Positive Breakout, "Unverifiable" = Cannot be verified)
        """
        # Boundary judgment: insufficient subsequent data to verify at the end of the signal series
        if signal_idx + self.holding_period >= len(self.prices):
            return "Unverifiable"

        # Extract core data
        signal_price = self.prices.iloc[signal_idx]
        trendline_price = self.downtrend_line[signal_idx]
        post_signal_prices = self.prices.iloc[signal_idx: signal_idx + self.holding_period + 1]

        # Calculate verification metrics
        post_high = post_signal_prices.max()
        post_low = post_signal_prices.min()
        max_gain = (post_high - signal_price) / signal_price  # Maximum gain (decimal form)
        is_support_valid = (post_low >= trendline_price)  # Whether resistance-to-support conversion is effective

        # True/false judgment (quantitative standards for large-cap tech stocks)
        if max_gain >= self.min_threshold and is_support_valid:
            return "TPB"
        elif max_gain < self.min_threshold and post_low < trendline_price:
            return "FPB"
        else:
            return "Unverifiable"

    def _verify_breakdown_single(self, signal_idx: int) -> str:
        """
        Helper Method: Verify true/false of a single candidate breakdown signal (holding period + price change threshold).
        :param signal_idx: Signal index
        :return: Verification result ("TPBd" = True Positive Breakdown, "FPBd" = False Positive Breakdown, "Unverifiable" = Cannot be verified)
        """
        # Boundary judgment: insufficient subsequent data to verify at the end of the signal series
        if signal_idx + self.holding_period >= len(self.prices):
            return "Unverifiable"

        # Extract core data
        signal_price = self.prices.iloc[signal_idx]
        trendline_price = self.uptrend_line[signal_idx]
        post_signal_prices = self.prices.iloc[signal_idx: signal_idx + self.holding_period + 1]

        # Calculate verification metrics
        post_low = post_signal_prices.min()
        post_high = post_signal_prices.max()
        max_loss = (signal_price - post_low) / signal_price  # Maximum loss (decimal form)
        is_resistance_valid = (post_high <= trendline_price)  # Whether support-to-resistance conversion is effective

        # True/false judgment (quantitative standards for large-cap tech stocks)
        if max_loss >= self.min_threshold and is_resistance_valid:
            return "TPBd"
        elif max_loss < self.min_threshold and post_high > trendline_price:
            return "FPBd"
        else:
            return "Unverifiable"

    def _calculate_performance_metrics(self, true_positive: int, false_positive: int, false_negative: int) -> dict:
        """
        Helper Method: Calculate classification performance metrics (Precision, Recall, F1-Score, False Signal Rate).
        :param true_positive: Number of true positives
        :param false_positive: Number of false positives
        :param false_negative: Number of false negatives
        :return: Dictionary of performance metrics
        """
        # Avoid division by zero (metrics are 0 when there are no signals)
        total_detected = true_positive + false_positive
        total_actual = true_positive + false_negative

        # Calculate core metrics
        precision = true_positive / total_detected if total_detected > 0 else 0.0
        recall = true_positive / total_actual if total_actual > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        false_signal_rate = false_positive / total_detected if total_detected > 0 else 0.0

        # Format results (retain 4 decimal places for better readability)
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1_score, 4),
            "false_signal_rate": round(false_signal_rate, 4)
        }

    def evaluate_signals(self, analyzer_result: dict, false_negative_breakout: int = 0, false_negative_breakdown: int = 0) -> dict:
        """
        Core Method: Batch verify all candidate signals and calculate performance metrics.
        :param analyzer_result: Analysis result dictionary from StockAnalyzer
        :param false_negative_breakout: Number of false negative breakouts (missed valid signals, manually labeled)
        :param false_negative_breakdown: Number of false negative breakdowns (missed valid signals, manually labeled)
        :return: Complete evaluation result dictionary
        """
        # Extract core data from analysis results (initialize class data)
        self.prices = analyzer_result["data"]["prices"]
        self.volume = analyzer_result["data"]["volume"]
        self.uptrend_line = analyzer_result["trendlines"]["uptrend_line"]
        self.downtrend_line = analyzer_result["trendlines"]["downtrend_line"]
        breakout_indices = analyzer_result["signals"]["breakout_indices"]
        breakdown_indices = analyzer_result["signals"]["breakdown_indices"]

        # Batch verify breakout signals
        self.breakout_verification = [self._verify_breakout_single(idx) for idx in breakout_indices]

        # Batch verify breakdown signals
        self.breakdown_verification = [self._verify_breakdown_single(idx) for idx in breakdown_indices]

        # Statistic basic metrics
        breakout_stats = {
            "TPB": self.breakout_verification.count("TPB"),
            "FPB": self.breakout_verification.count("FPB"),
            "FNB": false_negative_breakout,
            "TNB": len(self.prices) - len(breakout_indices) - false_negative_breakout - self.breakout_verification.count("Unverifiable"),
            "unverifiable": self.breakout_verification.count("Unverifiable")
        }

        breakdown_stats = {
            "TPBd": self.breakdown_verification.count("TPBd"),
            "FPBd": self.breakdown_verification.count("FPBd"),
            "FNBd": false_negative_breakdown,
            "TNBd": len(self.prices) - len(breakdown_indices) - false_negative_breakdown - self.breakdown_verification.count("Unverifiable"),
            "unverifiable": self.breakdown_verification.count("Unverifiable")
        }

        # Calculate performance metrics (Precision, Recall, F1-Score)
        breakout_derived = self._calculate_performance_metrics(
            breakout_stats["TPB"], breakout_stats["FPB"], breakout_stats["FNB"]
        )

        breakdown_derived = self._calculate_performance_metrics(
            breakdown_stats["TPBd"], breakdown_stats["FPBd"], breakdown_stats["FNBd"]
        )

        # Organize final results
        self.evaluation_result = {
            "breakout_stats": breakout_stats,
            "breakdown_stats": breakdown_stats,
            "breakout_derived_metrics": breakout_derived,
            "breakdown_derived_metrics": breakdown_derived,
            "verification_details": {
                "holding_period": self.holding_period,
                "min_threshold": self.min_threshold,
                "breakout_verification": self.breakout_verification,
                "breakdown_verification": self.breakdown_verification
            }
        }

        print("Signal evaluation completed. All performance metrics have been calculated.")
        return self.evaluation_result

    def export_to_excel(self, excel_path: str) -> None:
        """
        Core Method: Export evaluation results to Excel file (facilitate subsequent analysis and report writing).
        :param excel_path: Excel file save path
        """
        if not self.evaluation_result:
            raise ValueError("Evaluation result is empty. Please call evaluate_signals() to complete signal evaluation first.")

        # Write to Excel file, store results in 3 worksheets
        try:
            with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                # Worksheet 1: Breakout signal statistics
                breakout_df = pd.DataFrame.from_dict(self.evaluation_result["breakout_stats"], orient="index", columns=["Value"])
                breakout_df.to_excel(writer, sheet_name="Breakout_Stats")

                # Worksheet 2: Breakdown signal statistics
                breakdown_df = pd.DataFrame.from_dict(self.evaluation_result["breakdown_stats"], orient="index", columns=["Value"])
                breakdown_df.to_excel(writer, sheet_name="Breakdown_Stats")

                # Worksheet 3: Derived performance metrics
                derived_df = pd.DataFrame({
                    "Breakout": self.evaluation_result["breakout_derived_metrics"].values(),
                    "Breakdown": self.evaluation_result["breakdown_derived_metrics"].values()
                }, index=self.evaluation_result["breakout_derived_metrics"].keys())
                derived_df.to_excel(writer, sheet_name="Derived_Metrics")

            print(f"Evaluation results have been successfully exported to Excel file: {excel_path}")
        except Exception as e:
            raise Exception(f"Failed to export Excel file: {str(e)}")