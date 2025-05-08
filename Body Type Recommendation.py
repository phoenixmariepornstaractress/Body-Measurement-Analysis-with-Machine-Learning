# Body Type Recommendation System
"""
This module provides personalized clothing and fitness recommendations based on common body types.
It includes utilities for recommendation retrieval, keyword-based search, and comparison between body types.
"""

# Body type recommendation data
body_type_recommendations = {
    "Hourglass": {
        "clothing": [
            "Fitted dresses that highlight the waist",
            "Wrap tops and belted jackets",
            "High-waisted trousers or skirts"
        ],
        "fitness": [
            "Full-body strength training",
            "Core stabilization exercises",
            "HIIT cardio sessions"
        ]
    },
    "Pear": {
        "clothing": [
            "A-line skirts and wide-leg pants",
            "Structured tops with bold shoulders",
            "Light-colored tops and dark bottoms"
        ],
        "fitness": [
            "Lower body strength (e.g., lunges, squats)",
            "Core work for posture",
            "Cardio focused on fat loss"
        ]
    },
    "Apple": {
        "clothing": [
            "Empire waist dresses",
            "V-neck and scoop necklines",
            "Flowy tops with fitted pants"
        ],
        "fitness": [
            "Cardio to reduce central fat",
            "Strength training to build overall muscle",
            "Pilates or yoga for core strength"
        ]
    },
    "Rectangle": {
        "clothing": [
            "Peplum tops and ruffled blouses",
            "Cinched-waist jackets",
            "Layered outfits to add curves"
        ],
        "fitness": [
            "Targeted glute and chest exercises",
            "Abdominal work for tone",
            "Resistance training for definition"
        ]
    },
    "Inverted Triangle": {
        "clothing": [
            "Flared pants and A-line skirts",
            "V-necks and lower necklines",
            "Soft fabrics that drape below the waist"
        ],
        "fitness": [
            "Lower-body strength workouts",
            "Core exercises for stability",
            "Limit upper-body overload"
        ]
    }
}

# Core Functions
def get_recommendations(body_type):
    """Return clothing and fitness suggestions for a given body type."""
    return body_type_recommendations.get(
        body_type,
        {"clothing": ["No recommendation available."], "fitness": ["No recommendation available."]}
    )

def is_supported_body_type(body_type):
    """Check if a given body type is supported."""
    return body_type in body_type_recommendations

def list_supported_body_types():
    """Return a list of all supported body types."""
    return list(body_type_recommendations.keys())

def print_recommendations(body_type):
    """Print clothing and fitness suggestions in a readable format."""
    recommendations = get_recommendations(body_type)
    print(f"\nRecommendations for {body_type} Body Type:")
    print("\nClothing Suggestions:")
    for item in recommendations["clothing"]:
        print(f"- {item}")
    print("\nFitness Suggestions:")
    for item in recommendations["fitness"]:
        print(f"- {item}")

# Advanced Utility Functions
def search_recommendations_by_keyword(keyword):
    """Search for body types that include a specific keyword in clothing or fitness tips."""
    matches = []
    for body_type, recs in body_type_recommendations.items():
        combined = recs["clothing"] + recs["fitness"]
        if any(keyword.lower() in item.lower() for item in combined):
            matches.append(body_type)
    return matches

def compare_body_types(type1, type2):
    """Compare clothing and fitness recommendations between two body types."""
    if not (is_supported_body_type(type1) and is_supported_body_type(type2)):
        return {"error": "Comparison failed. One or both body types are unsupported."}

    rec1 = get_recommendations(type1)
    rec2 = get_recommendations(type2)

    return {
        "type1": type1,
        "type2": type2,
        "clothing_difference": list(set(rec1["clothing"]) ^ set(rec2["clothing"])),
        "fitness_difference": list(set(rec1["fitness"]) ^ set(rec2["fitness"]))
    }

# Example Usage
def main():
    predicted_body_type = "Hourglass"
    if is_supported_body_type(predicted_body_type):
        print_recommendations(predicted_body_type)
    else:
        print(f"No recommendations found for body type: {predicted_body_type}")

    print("\nSupported Body Types:")
    for btype in list_supported_body_types():
        print(f"- {btype}")

    print("\nBody types that mention 'core':")
    matched = search_recommendations_by_keyword("core")
    for match in matched:
        print(f"- {match}")

    print("\nComparison between Hourglass and Rectangle:")
    comparison = compare_body_types("Hourglass", "Rectangle")
    if "error" in comparison:
        print(comparison["error"])
    else:
        print("Clothing Differences:", comparison["clothing_difference"])
        print("Fitness Differences:", comparison["fitness_difference"])

if __name__ == "__main__":
    main()
