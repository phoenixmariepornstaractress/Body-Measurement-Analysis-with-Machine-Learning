# A professional Python script to represent and utilize body measurements and body type in a structured format.
# This could be used for fashion tech applications, character modeling, or health & fitness tracking systems.

from dataclasses import dataclass
from typing import Optional
import numpy as np
import math
import re

@dataclass
class BodyMeasurements:
    bust: str   # Example: '32GG'
    waist: int  # In inches
    hips: int   # In inches
    butt_type: Optional[str] = None

    def _parse_bust_size(self) -> float:
        match = re.match(r"(\d+)([A-Z]+)", self.bust.upper())
        if not match:
            return 0.0
        band_size = int(match.group(1))
        cup_size = match.group(2)
        cup_map = {cup: i for i, cup in enumerate(
            ["A", "B", "C", "D", "DD", "E", "F", "FF", "G", "GG", "H", "HH", "J", "JJ", "K"], start=1)}
        cup_value = cup_map.get(cup_size, 0)
        return band_size + np.clip(math.sqrt(cup_value) * 2.0, 0, 20)

    def get_shape_profile(self) -> str:
        if self.bust and self.waist and self.hips:
            bust_numeric = self._parse_bust_size()
            hip_waist_ratio = self.hips / self.waist
            bust_waist_ratio = bust_numeric / self.waist

            if hip_waist_ratio >= 1.25 and bust_waist_ratio >= 1.25:
                return "Hourglass"
            elif hip_waist_ratio >= 1.3:
                return "Pear"
            elif bust_waist_ratio >= 1.3:
                return "Inverted Triangle"
            else:
                return "Rectangle"
        return "Unknown"

    def describe(self) -> str:
        return (f"Measurements: {self.bust}-{self.waist}-{self.hips}\n"
                f"Butt Type: {self.butt_type or 'Not Specified'}\n"
                f"Body Shape: {self.get_shape_profile()}")

    def body_mass_index(self, height_in_inches: float, age: int = 30) -> Optional[float]:
        if height_in_inches <= 0 or self.waist <= 0 or age <= 0:
            return None
        height_cm = height_in_inches * 2.54
        bmi = (1.20 * ((self.waist / height_cm) ** 2) * 10000) + (0.23 * age) - 5.4
        return round(bmi, 2)

    def waist_to_hip_ratio(self) -> Optional[float]:
        if self.waist > 0 and self.hips > 0:
            return round(self.waist / self.hips, 2)
        return None

    def bust_to_waist_ratio(self) -> Optional[float]:
        bust = self._parse_bust_size()
        return round(bust / self.waist, 2) if self.waist > 0 else None

    def hip_to_bust_ratio(self) -> Optional[float]:
        bust = self._parse_bust_size()
        return round(self.hips / bust, 2) if bust > 0 else None

    def is_curvy(self) -> bool:
        return self.get_shape_profile() in ["Hourglass", "Pear"]

    def is_proportional(self, tolerance: float = 0.1) -> bool:
        bust = self._parse_bust_size()
        return abs(bust - self.hips) / max(bust, self.hips) <= tolerance

    def recommendation(self) -> str:
        shape = self.get_shape_profile()
        recs = {
            "Hourglass": "Fitted dresses and belts to accentuate the waist.",
            "Pear": "A-line skirts and embellished tops to balance proportions.",
            "Inverted Triangle": "Wide-leg pants and V-neck tops to soften shoulders.",
            "Rectangle": "Peplum tops and ruching to create curves."
        }
        return recs.get(shape, "Choose clothing that makes you feel confident and comfortable.")

    def estimated_body_fat_percentage(self, age: int, gender: str = "female") -> Optional[float]:
        if gender.lower() != "female" or self.waist <= 0 or self.hips <= 0:
            return None
        ratio = self.waist_to_hip_ratio()
        return round(76 - (20 * ratio), 2) if ratio else None

    def symmetry_index(self) -> Optional[float]:
        bust = self._parse_bust_size()
        return round(abs(bust - self.hips) / max(bust, self.hips), 2) if self.hips > 0 and bust > 0 else None

    def size_category(self) -> str:
        bust = self._parse_bust_size()
        if bust < 34:
            return "Petite"
        elif 34 <= bust <= 38:
            return "Medium"
        else:
            return "Full-figured"

    def balance_score(self) -> Optional[float]:
        bust = self._parse_bust_size()
        if self.waist and self.hips and bust:
            avg_diff = np.mean([abs(bust - self.waist), abs(self.hips - self.waist)])
            return round(1 - (avg_diff / max(bust, self.hips)), 2)
        return None

    def total_volume_index(self) -> Optional[float]:
        bust = self._parse_bust_size()
        return round((bust + self.waist + self.hips) / 3, 2) if bust and self.waist and self.hips else None

    def shape_symmetry_ratio(self) -> Optional[float]:
        bust = self._parse_bust_size()
        return round(min(bust, self.hips) / max(bust, self.hips), 2) if bust and self.hips else None

    def silhouette_type(self) -> str:
        score = self.balance_score()
        if score is None:
            return "Undefined"
        elif score > 0.9:
            return "Balanced Silhouette"
        elif score > 0.75:
            return "Mildly Asymmetric"
        return "Highly Asymmetric"

    def body_proportion_score(self) -> Optional[float]:
        bust = self._parse_bust_size()
        return round((bust / self.waist + self.hips / self.waist) / 2, 2) if bust and self.waist and self.hips else None

    def fashion_fit_index(self) -> Optional[float]:
        scores = {"Hourglass": 0.95, "Pear": 0.85, "Inverted Triangle": 0.80, "Rectangle": 0.75}
        return scores.get(self.get_shape_profile(), 0.70)

    def is_idealized_proportions(self) -> Optional[bool]:
        bust = self._parse_bust_size()
        return abs(bust - self.hips) < 2 and self.waist < bust * 0.75

    def bust_hip_difference(self) -> float:
        return round(abs(self._parse_bust_size() - self.hips), 2)

    def is_balanced_body(self) -> bool:
        sym = self.symmetry_index()
        return sym is not None and sym <= 0.1

    def enhancement_focus(self) -> str:
        bust = self._parse_bust_size()
        if self.hips > bust:
            return "Upper body enhancement (e.g., padded bras, shoulder details)"
        elif bust > self.hips:
            return "Lower body enhancement (e.g., padded hips, flared skirts)"
        return "Balanced body – no enhancement needed"

    def curvature_balance_index(self) -> Optional[float]:
        bust = self._parse_bust_size()
        if self.waist == 0:
            return None
        return round(((bust - self.waist) + (self.hips - self.waist)) / (2 * self.waist), 2)

    def hourglass_similarity_score(self) -> Optional[float]:
        bust = self._parse_bust_size()
        if not all([bust, self.waist, self.hips]):
            return None
        ideal = np.array([36, 24, 36])
        actual = np.array([bust, self.waist, self.hips])
        return round(1 / (1 + np.linalg.norm(actual - ideal)), 3)

if __name__ == "__main__":
    user_profile = BodyMeasurements(bust="32GG", waist=28, hips=36, butt_type="Bubble Butt")
    print(user_profile.describe())
    print(f"BMI (adjusted): {user_profile.body_mass_index(65)}")
    print(f"Waist-to-Hip Ratio: {user_profile.waist_to_hip_ratio()}")
    print(f"Bust-to-Waist Ratio: {user_profile.bust_to_waist_ratio()}")
    print(f"Hip-to-Bust Ratio: {user_profile.hip_to_bust_ratio()}")
    print(f"Is Curvy: {'Yes' if user_profile.is_curvy() else 'No'}")
    print(f"Is Proportional: {'Yes' if user_profile.is_proportional() else 'No'}")
    print(f"Style Recommendation: {user_profile.recommendation()}")
    print(f"Estimated Body Fat %: {user_profile.estimated_body_fat_percentage(age=30)}")
    print(f"Symmetry Index: {user_profile.symmetry_index()}")
    print(f"Size Category: {user_profile.size_category()}")
    print(f"Balance Score: {user_profile.balance_score()}")
    print(f"Total Volume Index: {user_profile.total_volume_index()}")
    print(f"Shape Symmetry Ratio: {user_profile.shape_symmetry_ratio()}")
    print(f"Silhouette Type: {user_profile.silhouette_type()}")
    print(f"Body Proportion Score: {user_profile.body_proportion_score()}")
    print(f"Fashion Fit Index: {user_profile.fashion_fit_index()}")
    print(f"Idealized Proportions: {'Yes' if user_profile.is_idealized_proportions() else 'No'}")
    print(f"Bust-Hip Difference: {user_profile.bust_hip_difference()}")
    print(f"Is Balanced Body: {'Yes' if user_profile.is_balanced_body() else 'No'}")
    print(f"Enhancement Focus: {user_profile.enhancement_focus()}")
    print(f"Curvature Balance Index: {user_profile.curvature_balance_index()}")
    print(f"Hourglass Similarity Score: {user_profile.hourglass_similarity_score()}")
