# Organized Seaborn-based Visualization for Body Measurement Analysis

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from body_measurement_analysis import BodyMeasurements

# Sample data for visualization
sample_data = [
    {"bust": "32GG", "waist": 28, "hips": 36},
    {"bust": "34D", "waist": 26, "hips": 37},
    {"bust": "36C", "waist": 29, "hips": 39},
    {"bust": "30E", "waist": 24, "hips": 35},
    {"bust": "38F", "waist": 32, "hips": 42}
]

# Generate DataFrame from sample data
def create_dataframe(data):
    records = []
    for entry in data:
        person = BodyMeasurements(bust=entry["bust"], waist=entry["waist"], hips=entry["hips"])
        records.append({
            "Bust (in)": person.bust_measurement(),
            "Waist (in)": person.waist,
            "Hips (in)": person.hips,
            "WHR": person.waist_to_hip_ratio(),
            "Bust-to-Waist": person.bust_to_waist_ratio(),
            "Hips-to-Waist": person.hips_to_waist_ratio(),
            "Body Type": person.body_type(),
            "Figure Grade": person.figure_grade(),
            "Health Category": person.health_category(),
            "Shape Index": person.shape_index(),
            "Symmetry Score": person.symmetry_score(),
            "Dominant Feature": person.dominant_feature(),
            "Overall Balance": person.overall_balance(),
            "Proportion Score": person.proportion_score(),
            "Style Compatibility": person.style_compatibility(),
            "Posture Index": person.posture_index(),
            "Curve Ratio": person.curve_ratio(),
            "Size Consistency": person.size_consistency()
        })
    return pd.DataFrame(records)

# Create the DataFrame
df = create_dataframe(sample_data)

# Set Seaborn theme
sns.set_theme(style="whitegrid")

# --- Visualization Section ---

# Box Plot: Basic measurements
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[["Bust (in)", "Waist (in)", "Hips (in)"]])
plt.title("Box Plot of Body Measurements")
plt.show()

# Scatter Plot: Bust vs. Hips
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="Bust (in)", y="Hips (in)", hue="Body Type", style="Figure Grade", s=100)
plt.title("Bust vs. Hips by Body Type and Figure Grade")
plt.xlabel("Bust (inches)")
plt.ylabel("Hips (inches)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Bar Plot: WHR by Health Category
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x="Health Category", y="WHR", palette="muted")
plt.title("Waist-to-Hip Ratio by Health Category")
plt.tight_layout()
plt.show()

# Violin Plot: Symmetry Score by Figure Grade
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x="Figure Grade", y="Symmetry Score", palette="coolwarm")
plt.title("Symmetry Score by Figure Grade")
plt.tight_layout()
plt.show()

# Count Plot: Dominant Features
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x="Dominant Feature", palette="pastel")
plt.title("Dominant Body Features")
plt.tight_layout()
plt.show()

# Bar Plot: Overall Balance by Body Type
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x="Body Type", y="Overall Balance", palette="Blues_d")
plt.title("Overall Balance Across Body Types")
plt.tight_layout()
plt.show()

# Line Plot: Proportion Score vs Bust
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="Bust (in)", y="Proportion Score", marker="o")
plt.title("Proportion Score by Bust Measurement")
plt.tight_layout()
plt.show()

# Bar Plot: Style Compatibility
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x="Figure Grade", y="Style Compatibility", palette="Greens")
plt.title("Style Compatibility by Figure Grade")
plt.tight_layout()
plt.show()

# Scatter Plot: Posture Index vs Symmetry Score
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="Posture Index", y="Symmetry Score", hue="Body Type", s=100)
plt.title("Posture Index vs Symmetry Score")
plt.tight_layout()
plt.show()

# Bar Plot: Curve Ratio by Dominant Feature
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x="Dominant Feature", y="Curve Ratio", palette="Purples")
plt.title("Curve Ratio by Dominant Feature")
plt.tight_layout()
plt.show()

# Box Plot: Size Consistency
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="Body Type", y="Size Consistency", palette="Oranges")
plt.title("Size Consistency by Body Type")
plt.tight_layout()
plt.show()

# Pairplot: Overall visual correlations
sns.pairplot(
    df,
    hue="Body Type",
    vars=[
        "Bust (in)", "Waist (in)", "Hips (in)",
        "WHR", "Symmetry Score", "Proportion Score", "Posture Index"
    ]
)
plt.suptitle("Pairwise Comparison of Body Measurements", y=1.02)
plt.show()
