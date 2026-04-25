import os
import json
from google import genai
from dotenv import load_dotenv

# Resolve project root from this file's location (src/ -> project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load GEMINI_API_KEY from the .env file in the project root
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# Use the current stable Gemini Flash model
GEMINI_MODEL = "gemini-2.0-flash"


def _get_client():
    """Validates the API key and returns a configured Gemini client."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "paste_your_key_here":
        raise ValueError(
            "GEMINI_API_KEY is not set. Please add your key to the .env file."
        )
    return genai.Client(api_key=api_key)


def get_retention_strategy(top_features, churn_rate, high_risk_count, avg_revenue):
    """
    Generates a plain-English 3-point retention strategy using the Gemini API.

    Args:
        top_features    (dict):  Top 5 churn driver features and their importance scores.
        churn_rate      (float): Overall churn rate as a percentage (e.g. 26.54).
        high_risk_count (int):   Number of customers flagged as high-risk.
        avg_revenue     (float): Average monthly revenue per customer in dollars.

    Returns:
        str: A 3-point retention strategy in plain English, or a fallback string on error.
    """
    try:
        client = _get_client()

        # Format the top features into a readable list for the prompt
        feature_lines = "\n".join(
            f"  - {feat}: importance score {score:.4f}"
            for feat, score in top_features.items()
        )

        prompt = f"""
You are a senior telecom retention strategist presenting findings to a CFO.

Here are the machine learning results from our customer churn model:

Churn rate: {churn_rate:.2f}%
High-risk customers identified: {high_risk_count}
Average monthly revenue per customer: ${avg_revenue:.2f}

Top churn driver features:
{feature_lines}

Write a 3-point retention strategy. Each point must:
1. Name the specific churn driver from the list above
2. Propose one concrete intervention the business can execute immediately
3. State the expected outcome in measurable business terms

Write in plain text with no markdown formatting, no bullet symbols, no asterisks, and no headers.
Keep the total response under 120 words. Use a confident, executive-facing tone.
"""

        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return response.text.strip()

    except Exception as e:
        return (
            f"[Gemini API Error] Could not generate retention strategy: {str(e)}. "
            "Please check your GEMINI_API_KEY in the .env file."
        )


def get_customer_risk_explanation(customer_data, churn_probability):
    """
    Generates a 2-sentence plain-English explanation of why a specific customer
    is at risk of churning and what single action to take.

    Args:
        customer_data     (dict):  A dictionary of the customer's feature values.
        churn_probability (float): The model's predicted churn probability (0.0 to 1.0).

    Returns:
        str: Two sentences — one explaining the risk, one recommending an action.
             Returns a fallback string on API error.
    """
    try:
        client = _get_client()

        # Format customer data as readable key-value pairs for the prompt
        customer_lines = "\n".join(
            f"  {key}: {value}" for key, value in customer_data.items()
        )

        prompt = f"""
You are a customer retention analyst. You have been given the profile of a single telecom customer
and their predicted probability of cancelling their subscription.

Churn probability: {churn_probability * 100:.1f}%

Customer profile:
{customer_lines}

Write exactly 2 sentences in plain text with no markdown, no bullet points, and no asterisks:
Sentence 1: Explain the single most likely reason this specific customer will churn, based on their data.
Sentence 2: Recommend the single most effective retention action for this specific customer.
"""

        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return response.text.strip()

    except Exception as e:
        return (
            f"[Gemini API Error] Could not generate customer explanation: {str(e)}. "
            "Please check your GEMINI_API_KEY in the .env file."
        )


if __name__ == "__main__":
    """
    Quick smoke test — loads the real top_features.json from the models/ directory
    and calls get_retention_strategy with sample business numbers to verify the API works.
    """
    print("Loading top features from models/top_features.json...")
    top_features_path = os.path.join(PROJECT_ROOT, "models", "top_features.json")

    with open(top_features_path, "r") as f:
        top_features = json.load(f)

    print(f"Top features loaded: {list(top_features.keys())}\n")
    print("Calling Gemini API — get_retention_strategy()...\n")

    result = get_retention_strategy(
        top_features=top_features,
        churn_rate=26.54,
        high_risk_count=412,
        avg_revenue=64.76,
    )

    print("=" * 60)
    print("GEMINI RETENTION STRATEGY OUTPUT:")
    print("=" * 60)
    print(result)
    print("=" * 60)
