
import os
from dotenv import load_dotenv

load_dotenv()


def gemini_available():
    """Check whether a Gemini API key is configured."""
    return bool(os.getenv("GEMINI_API_KEY"))


def generate_advice(lead, score_result):
    """
    Generate AI-based sales advice.
    If Gemini is unavailable, use the local recommendation.
    """

    # Prepare a reliable local fallback
    recommendation = score_result.get(
        "recommendation",
        "Review the lead and understand its business requirements."
    )

    fallback = (
        "AI Recommendation (Local Engine)\n\n"
        f"{recommendation}\n\n"
        "Note: The lead score is an estimate, not a guarantee "
        "of conversion. A sales representative should review "
        "the lead before taking action."
    )

    # Check API key
    key = os.getenv("GEMINI_API_KEY")

    if not key:
        return fallback

    try:
        # Import Google Gemini SDK
        from google import genai
        from google.genai import types

        # Initialize Gemini client
        client = genai.Client(
            api_key=key,
            http_options=types.HttpOptions(
                timeout=10000
            )
        )

        # Get configured model
        model = os.getenv(
            "GEMINI_MODEL",
            "gemini-4-hydrogen"
        ).strip()

        # Prepare lead information
        prompt = f"""
You are a careful and professional sales-support assistant.

Your task is to provide useful, concise and ethical
recommendations to sales representatives.

Rules:
- Do not infer sensitive personal characteristics.
- Do not guarantee lead conversion.
- Do not contact customers automatically.
- A human must review all sales outreach.

LEAD INFORMATION:

Industry: {lead.get('industry', 'Not provided')}
Company: {lead.get('company', 'Not provided')}
Company Size: {lead.get('company_size', 'Not provided')}
Estimated Budget: {lead.get('budget', 0)}
Website Visits: {lead.get('website_visits', 0)}
Email Opens: {lead.get('email_opens', 0)}
Demo Requested: {lead.get('demo_requested', False)}

MODEL RESULTS:

Lead Score: {score_result.get('score', 0)}/100
Priority: {score_result.get('priority', 'Not available')}

Provide your response in three sections:

1. Lead Analysis:
Briefly explain the lead's engagement and potential.

2. Recommended Next Step:
Suggest one practical action for the sales representative.

3. Data Limitation:
Mention one uncertainty or missing information.

Keep the response professional and concise.
"""

        # Generate Gemini response
        response = client.models.generate_content(
            model=model,
            contents=prompt
        )

        # Validate response
        if response and response.text:
            result = response.text.strip()

            if result:
                return result

        # Empty response fallback
        return fallback

    except Exception:
        # Gracefully handle API, network and model errors
        return fallback
