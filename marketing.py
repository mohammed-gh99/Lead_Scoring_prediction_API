import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# Use localhost by default. Set PREDICTION_API_URL to the active AWS endpoint
# when the API is deployed to Elastic Beanstalk.
API_URL = os.getenv(
    "PREDICTION_API_URL",
    "http://localhost:9696/predict",
)

customer = {
    "lead_source": "organic_search",
    "industry": "technology",
    "employment_status": "employed",
    "location": "africa",
    "annual_income": 50000,
    "number_of_courses_viewed": 5,
    "interaction_count": 10,
    "lead_score": 80,
}


def predict_lead(customer_data: dict) -> dict:
    """Send one lead to the prediction API and return its JSON response."""
    payload = json.dumps(customer_data).encode("utf-8")
    req = Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(req, timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Prediction API returned HTTP {exc.code}: {error_body}"
        ) from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(
            f"Could not connect to the prediction API at {API_URL}: {exc}"
        ) from exc


def main() -> None:
    try:
        result = predict_lead(customer)
        probability = float(result["converted_probability"])
        converted = result["converted"]
    except (RuntimeError, KeyError, TypeError, ValueError) as exc:
        print(f"Prediction failed: {exc}")
        raise SystemExit(1) from exc

    print(f"Conversion probability: {probability:.2%}")
    print(f"Predicted conversion: {converted}")

    if converted:
        print("Action: Prioritize this lead for sales follow-up.")
    else:
        print("Action: Add this lead to a nurturing campaign.")


if __name__ == "__main__":
    main()
