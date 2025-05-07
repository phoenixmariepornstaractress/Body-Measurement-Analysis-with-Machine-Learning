# Professional Python code to model and analyze body measurements

import re
from typing import Tuple, Dict, List

class BodyMeasurements:
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
        base_cups = ['A', 'B', 'C', 'D', 'DD', 'E', 'F', 'FF', 'G', 'GG', 'H']
        _, cup = self.parse_bra_size()
        return base_cups.index(cup) if cup in base_cups else 0

    def bust_measurement(self) -> int:
        band_size, _ = self.parse_bra_size()
        return band_size + self.estimate_cup_volume()

    def waist_to_hip_ratio(self) -> float:
        return round(self.waist / self.hips, 2) if self.hips != 0 else 0.0

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
        return "Straight/Rectangle"

    def clothing_size_estimate(self) -> str:
        if self.waist <= 25:
            return "XS"
        elif 26 <= self.waist <= 27:
            return "S"
        elif 28 <= self.waist <= 29:
            return "M"
        elif 30 <= self.waist <= 32:
            return "L"
        return "XL or above"

    def hip_to_bust_ratio(self) -> float:
        bust = self.bust_measurement()
        return round(self.hips / bust, 2) if bust != 0 else 0.0

    def proportionality_score(self) -> str:
        ratio = self.hip_to_bust_ratio()
        if 0.9 <= ratio <= 1.1:
            return "Proportional"
        elif ratio < 0.9:
            return "Bust-dominant"
        return "Hip-dominant"

    def measurement_list(self) -> List[int]:
        return [self.bust_measurement(), self.waist, self.hips]

    def is_balanced(self) -> bool:
        bust, waist, hips = self.measurement_list()
        return abs(bust - hips) <= 2 and abs(bust - waist) > 6

    def total_measurement_volume(self) -> int:
        return sum(self.measurement_list())

    def average_measurement(self) -> float:
        return round(self.total_measurement_volume() / 3, 2)

    def dominant_feature(self) -> str:
        bust, waist, hips = self.measurement_list()
        max_val = max(bust, waist, hips)
        if max_val == bust:
            return "Bust"
        elif max_val == hips:
            return "Hips"
        return "Waist"

    def measurement_differences(self) -> Dict[str, int]:
        bust = self.bust_measurement()
        return {
            "Bust-Waist": bust - self.waist,
            "Hip-Waist": self.hips - self.waist,
            "Hip-Bust": self.hips - bust
        }

    def shape_index(self) -> float:
        bust, waist, hips = self.measurement_list()
        try:
            return round((bust * hips) / (waist ** 2), 2)
        except ZeroDivisionError:
            return 0.0

    def health_category(self) -> str:
        whr = self.waist_to_hip_ratio()
        if whr < 0.8:
            return "Low risk"
        elif 0.8 <= whr < 0.85:
            return "Moderate risk"
        return "High risk"

    def bust_to_waist_ratio(self) -> float:
        return round(self.bust_measurement() / self.waist, 2) if self.waist != 0 else 0.0

    def hips_to_waist_ratio(self) -> float:
        return round(self.hips / self.waist, 2) if self.waist != 0 else 0.0

    def symmetry_score(self) -> float:
        bust, _, hips = self.measurement_list()
        return round(1 - abs(bust - hips) / max(bust, hips), 2) if max(bust, hips) != 0 else 0.0

    def figure_grade(self) -> str:
        score = self.symmetry_score()
        if score > 0.9:
            return "Excellent Symmetry"
        elif score > 0.75:
            return "Good Symmetry"
        elif score > 0.6:
            return "Moderate Symmetry"
        return "Asymmetrical"

    def bust_proportion_of_total(self) -> float:
        bust = self.bust_measurement()
        total = self.total_measurement_volume()
        return round((bust / total) * 100, 2) if total != 0 else 0.0

    def waist_proportion_of_total(self) -> float:
        total = self.total_measurement_volume()
        return round((self.waist / total) * 100, 2) if total != 0 else 0.0

    def hips_proportion_of_total(self) -> float:
        total = self.total_measurement_volume()
        return round((self.hips / total) * 100, 2) if total != 0 else 0.0

    def volume_distribution(self) -> Dict[str, float]:
        return {
            "Bust %": self.bust_proportion_of_total(),
            "Waist %": self.waist_proportion_of_total(),
            "Hips %": self.hips_proportion_of_total()
        }

    def symmetry_variance(self) -> float:
        bust, _, hips = self.measurement_list()
        return round(abs(bust - hips) / ((bust + hips) / 2), 2) if (bust + hips) != 0 else 0.0

    def overall_balance(self) -> str:
        variance = self.symmetry_variance()
        if variance < 0.1:
            return "Well-balanced"
        elif variance < 0.2:
            return "Slightly unbalanced"
        return "Unbalanced"

    def summary(self) -> Dict[str, str]:
        band_size, cup_size = self.parse_bra_size()
        differences = self.measurement_differences()
        proportions = self.volume_distribution()
        return {
            "Bust": f"{band_size}{cup_size} ({self.bust_measurement()} in)",
            "Waist": f"{self.waist} in",
            "Hips": f"{self.hips} in",
            "Body Type": self.body_type(),
            "Waist-to-Hip Ratio": str(self.waist_to_hip_ratio()),
            "Ideal WHR": "Yes" if self.is_ideal_waist_to_hip() else "No",
            "Estimated Clothing Size": self.clothing_size_estimate(),
            "Hip-to-Bust Ratio": str(self.hip_to_bust_ratio()),
            "Proportionality": self.proportionality_score(),
            "Balanced Figure": "Yes" if self.is_balanced() else "No",
            "Total Volume": f"{self.total_measurement_volume()} in",
            "Average Measurement": f"{self.average_measurement()} in",
            "Dominant Feature": self.dominant_feature(),
            "Bust-Waist Difference": f"{differences['Bust-Waist']} in",
            "Hip-Waist Difference": f"{differences['Hip-Waist']} in",
            "Hip-Bust Difference": f"{differences['Hip-Bust']} in",
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
        return "\n".join(f"{k}: {v}" for k, v in self.summary().items())

# Example usage:
if __name__ == "__main__":
    measurements = BodyMeasurements(bust="32GG", waist=28, hips=36)
    print(measurements.descriptive_summary())
