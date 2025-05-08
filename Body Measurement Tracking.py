# Body Measurement Time-Series Tracking
# Track changes in body measurements over time for fitness or health monitoring.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class BodyMeasurementTracker:
    def __init__(self):
        self.data = pd.DataFrame(columns=["date", "bust", "waist", "hips"])

    def add_entry(self, date, bust, waist, hips):
        new_entry = {
            "date": pd.to_datetime(date),
            "bust": bust,
            "waist": waist,
            "hips": hips
        }
        self.data = pd.concat([self.data, pd.DataFrame([new_entry])], ignore_index=True)
        self.data.sort_values("date", inplace=True)

    def get_measurements(self):
        return self.data.copy()

    def plot_progress(self):
        if self.data.empty:
            print("No data available to plot.")
            return

        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 6))
        for col in ["bust", "waist", "hips"]:
            sns.lineplot(data=self.data, x="date", y=col, label=col.capitalize())
        plt.title("Body Measurement Progress Over Time")
        plt.xlabel("Date")
        plt.ylabel("Measurement (cm)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def calculate_bmi(self, height_cm):
        if self.data.empty:
            return None
        latest = self.data.iloc[-1]
        weight_estimate = (latest['bust'] + latest['waist'] + latest['hips']) / 3 * 0.45
        height_m = height_cm / 100
        bmi = weight_estimate / (height_m ** 2)
        return round(bmi, 2)

    def export_to_csv(self, filename):
        self.data.to_csv(filename, index=False)
        print(f"Data exported to {filename}")

    def get_summary_statistics(self):
        return self.data.describe()

    def average_change_rate(self):
        if len(self.data) < 2:
            return None
        self.data["days"] = (self.data["date"] - self.data["date"].iloc[0]).dt.days
        rates = {}
        for col in ["bust", "waist", "hips"]:
            delta = self.data[col].iloc[-1] - self.data[col].iloc[0]
            duration = self.data["days"].iloc[-1]
            rates[col] = round(delta / duration, 4) if duration != 0 else 0
        self.data.drop(columns="days", inplace=True)
        return rates

    def get_latest_entry(self):
        if self.data.empty:
            return None
        return self.data.iloc[-1]

    def trend_direction(self):
        trends = {}
        if len(self.data) < 2:
            return trends
        for col in ["bust", "waist", "hips"]:
            diff = self.data[col].iloc[-1] - self.data[col].iloc[0]
            if diff > 0:
                trends[col] = "increasing"
            elif diff < 0:
                trends[col] = "decreasing"
            else:
                trends[col] = "stable"
        return trends

    def percentage_change(self):
        if len(self.data) < 2:
            return None
        pct_changes = {}
        for col in ["bust", "waist", "hips"]:
            initial = self.data[col].iloc[0]
            final = self.data[col].iloc[-1]
            if initial != 0:
                pct_changes[col] = round(((final - initial) / initial) * 100, 2)
            else:
                pct_changes[col] = None
        return pct_changes

if __name__ == "__main__":
    tracker = BodyMeasurementTracker()

    # Add entries
    tracker.add_entry("2024-01-01", bust=92, waist=70, hips=96)
    tracker.add_entry("2024-03-01", bust=93, waist=68, hips=95)
    tracker.add_entry("2024-05-01", bust=94, waist=66, hips=94)

    # Show data
    print("\nCurrent Measurements:")
    print(tracker.get_measurements())

    # Plot measurements
    tracker.plot_progress()

    # Calculate BMI
    bmi = tracker.calculate_bmi(height_cm=165)
    print(f"\nEstimated BMI: {bmi}")

    # Export data
    tracker.export_to_csv("body_measurements.csv")

    # Summary stats
    print("\nSummary Statistics:")
    print(tracker.get_summary_statistics())

    # Average rate of change
    print("\nAverage Daily Change Rates:")
    print(tracker.average_change_rate())

    # Latest measurement
    print("\nLatest Entry:")
    print(tracker.get_latest_entry())

    # Trend analysis
    print("\nMeasurement Trends:")
    print(tracker.trend_direction())

    # Percent changes
    print("\nPercentage Changes:")
    print(tracker.percentage_change())
