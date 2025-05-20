import re
import math
import numpy as np
from typing import Tuple, Dict, List

class BodyMeasurements:
    CUP_SIZES = ['A', 'B', 'C', 'D', 'DD', 'E', 'F', 'FF', 'G', 'GG', 'H']

    def __init__(self, bust: str, waist: int, hips: int):
        self.bust = bust
        self.waist = waist
        self.hips = hips

    def parse_bra_size(self) -> Tuple[int, str]:
        match = re.match(r"(\d+)([A-Z]+)", self.bust.upper())
        if not match:
            raise ValueError("Invalid bra size format")
        return int(match.group(1)), match.group(2)

    def estimate_cup_volume(self) -> int:
        _, cup = self.parse_bra_size()
        return self.CUP_SIZES.index(cup) if cup in self.CUP_SIZES else 0

    def bust_measurement(self) -> int:
        band_size, _ = self.parse_bra_size()
        return band_size + self.estimate_cup_volume()

    def waist_to_hip_ratio(self) -> float:
        return round(self.waist / self.hips, 2) if self.hips else 0.0

    def is_ideal_waist_to_hip(self) -> bool:
        return self.waist_to_hip_ratio() < 0.8

    def body_type(self) -> str:
        bust_diff = self.bust_measurement() - self.waist
        hip_diff = self.hips - self.waist
        if bust_diff > 6 and hip_diff > 6:
            return "Hourglass"
        elif bust_diff > hip_diff:
            return "Top-heavy"
        elif hip_diff > bust_diff:
            return "Bottom-heavy"
        return "Rectangle"

    def clothing_size_estimate(self) -> str:
        if self.waist <= 25:
            return "XS"
        elif self.waist <= 27:
            return "S"
        elif self.waist <= 29:
            return "M"
        elif self.waist <= 32:
            return "L"
        return "XL+"

    def hip_to_bust_ratio(self) -> float:
        bust = self.bust_measurement()
        return round(self.hips / bust, 2) if bust else 0.0

    def proportionality_score(self) -> str:
        ratio = self.hip_to_bust_ratio()
        if 0.9 <= ratio <= 1.1:
            return "Proportional"
        return "Hip-dominant" if ratio > 1.1 else "Bust-dominant"

    def measurement_list(self) -> List[int]:
        return [self.bust_measurement(), self.waist, self.hips]

    def is_balanced(self) -> bool:
        bust, waist, hips = self.measurement_list()
        return abs(bust - hips) <= 2 and abs(bust - waist) > 6

    def total_measurement_volume(self) -> int:
        return sum(self.measurement_list())

    def average_measurement(self) -> float:
        return round(np.mean(self.measurement_list()), 2)

    def dominant_feature(self) -> str:
        values = self.measurement_list()
        features = ["Bust", "Waist", "Hips"]
        return features[values.index(max(values))]

    def measurement_differences(self) -> Dict[str, int]:
        bust = self.bust_measurement()
        return {
            "Bust-Waist": bust - self.waist,
            "Hip-Waist": self.hips - self.waist,
            "Hip-Bust": self.hips - bust
        }

    def shape_index(self) -> float:
        bust, waist, hips = self.measurement_list()
        return round((bust * hips) / (waist ** 2), 2) if waist else 0.0

    def health_category(self) -> str:
        whr = self.waist_to_hip_ratio()
        if whr < 0.8:
            return "Low risk"
        elif whr < 0.85:
            return "Moderate risk"
        return "High risk"

    def bust_to_waist_ratio(self) -> float:
        return round(self.bust_measurement() / self.waist, 2) if self.waist else 0.0

    def hips_to_waist_ratio(self) -> float:
        return round(self.hips / self.waist, 2) if self.waist else 0.0

    def symmetry_score(self) -> float:
        bust, _, hips = self.measurement_list()
        return round(1 - abs(bust - hips) / max(bust, hips), 2) if max(bust, hips) else 0.0

    def figure_grade(self) -> str:
        score = self.symmetry_score()
        if score > 0.9:
            return "Excellent"
        elif score > 0.75:
            return "Good"
        elif score > 0.6:
            return "Moderate"
        return "Asymmetrical"

    def volume_distribution(self) -> Dict[str, float]:
        total = self.total_measurement_volume()
        bust = self.bust_measurement()
        return {
            "Bust %": round((bust / total) * 100, 2) if total else 0.0,
            "Waist %": round((self.waist / total) * 100, 2) if total else 0.0,
            "Hips %": round((self.hips / total) * 100, 2) if total else 0.0
        }

    def symmetry_variance(self) -> float:
        bust, _, hips = self.measurement_list()
        return round(abs(bust - hips) / ((bust + hips) / 2), 2) if (bust + hips) else 0.0

    def overall_balance(self) -> str:
        variance = self.symmetry_variance()
        if variance < 0.1:
            return "Well-balanced"
        elif variance < 0.2:
            return "Slightly unbalanced"
        return "Unbalanced"

    def summary(self) -> Dict[str, str]:
        band_size, cup_size = self.parse_bra_size()
        diffs = self.measurement_differences()
        proportions = self.volume_distribution()
        return {
            "Bust": f"{band_size}{cup_size} ({self.bust_measurement()} in)",
            "Waist": f"{self.waist} in",
            "Hips": f"{self.hips} in",
            "Body Type": self.body_type(),
            "Waist-to-Hip Ratio": str(self.waist_to_hip_ratio()),
            "Ideal WHR": "Yes" if self.is_ideal_waist_to_hip() else "No",
            "Clothing Size": self.clothing_size_estimate(),
            "Hip-to-Bust Ratio": str(self.hip_to_bust_ratio()),
            "Proportionality": self.proportionality_score(),
            "Balanced Figure": "Yes" if self.is_balanced() else "No",
            "Total Volume": f"{self.total_measurement_volume()} in",
            "Average Measurement": f"{self.average_measurement()} in",
            "Dominant Feature": self.dominant_feature(),
            "Bust-Waist Difference": f"{diffs['Bust-Waist']} in",
            "Hip-Waist Difference": f"{diffs['Hip-Waist']} in",
            "Hip-Bust Difference": f"{diffs['Hip-Bust']} in",
            "Shape Index": str(self.shape_index()),
            "Health Category": self.health_category(),
            "Bust-to-Waist Ratio": str(self.bust_to_waist_ratio()),
            "Hips-to-Waist Ratio": str(self.hips_to_waist_ratio()),
            "Symmetry Score": str(self.symmetry_score()),
            "Figure Grade": self.figure_grade(),
            "Bust % of Total": f"{proportions['Bust %']}%",
            "Waist % of Total": f"{proportions['Waist %']}%",
            "Hips % of Total": f"{proportions['Hips %']}%",
            "Symmetry Variance": str(self.symmetry_variance()),
            "Overall Balance": self.overall_balance()
        }

    def descriptive_summary(self) -> str:
        return "\n".join(f"{key}: {value}" for key, value in self.summary().items())

if __name__ == "__main__":
    bm = BodyMeasurements(bust="32GG", waist=28, hips=36)
    print(bm.descriptive_summary())
 
