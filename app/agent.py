# ruff: noqa
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from typing import Any, AsyncGenerator
import google.auth
from google import genai
from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.events.event import Event
from google.adk.apps import App
from google.adk.workflow import Workflow

# Initialize GCP Environment Variables
_, project_id = google.auth.default()
os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"


def get_query_text(node_input: Any) -> str:
    """Helper to extract text query from various node input types."""
    if isinstance(node_input, types.Content):
        if node_input.parts:
            return "".join(part.text for part in node_input.parts if part.text)
    elif isinstance(node_input, dict):
        return node_input.get("query", str(node_input))
    return str(node_input)


def classify_query(node_input: Any) -> Event:
    """Classifies whether the user query is shipping-related or unrelated.

    Args:
        node_input: The input from the START node (types.Content or str)

    Returns:
        An Event containing the original query text and the determined route
        ('shipping' or 'unrelated').
    """
    query_text = get_query_text(node_input)

    # Initialize GenAI client to run within Vertex AI sandbox
    client = genai.Client()

    prompt = f"""You are a routing classifier for a shipping company's customer support.
Analyze the following user query and determine if it is related to shipping (rates, tracking, delivery, returns, shipping policies) or if it is unrelated (general knowledge, other topics, random chatter).

User Query: "{query_text}"

Respond in JSON with the following structure:
{{
  "is_shipping_related": true/false
}}
"""

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )
        data = json.loads(response.text)
        is_shipping = data.get("is_shipping_related", False)
    except Exception:
        # Graceful fallback in case of errors
        is_shipping = False

    route = "shipping" if is_shipping else "unrelated"
    return Event(output=query_text, actions={"route": route})


def decline_to_answer(node_input: str) -> AsyncGenerator[Event, None]:
    """Politely declines to answer queries unrelated to shipping.

    Args:
        node_input: The user query text.

    Yields:
        Event containing the model's text response for Web UI rendering.
        Event containing the node's final output.
    """
    text = (
        "I'm sorry, but I can only assist with shipping-related inquiries "
        "(such as shipping rates, package tracking, delivery status, and returns). "
        "How can I help you with your shipping needs today?"
    )
    yield Event(
        content=types.Content(role="model", parts=[types.Part.from_text(text=text)])
    )
    yield Event(output=text)


# Define the Shipping FAQ Agent
shipping_faq = LlmAgent(
    name="shipping_faq",
    model="gemini-3.5-flash",
    instruction="""You are a helpful customer support representative for a shipping company.
Your job is to answer the customer's query using the following official company policy and FAQ guidelines.
Be polite, professional, and helpful. Do not mention policies not listed here.

When responding to shipping rate queries, make them super playful, enthusiastic, and loaded with friendly emojis! 🎉✨ Always highlight the amazing FREE SHIPPING threshold of $50! 🛒🎁

---
OFFICIAL SHIPPING COMPANY POLICY & FAQs:

1. SHIPPING RATES:
- Standard Shipping: $5.99 (takes 3-5 business days). Free for orders over $50.
- Express Shipping: $14.99 (takes 1-2 business days).
- Overnight Shipping: $29.99 (takes next business day).

2. PACKAGE TRACKING:
- Customers can track packages on our website using their unique 10-digit tracking number (e.g. 1234567890).
- If tracking shows "delivered" but the package is missing, customers should verify with neighbors or contact tracking@shippingco.com.

3. DELIVERY:
- Delivery hours are Monday through Saturday, from 8:00 AM to 8:00 PM.
- A physical signature is required for all Overnight and high-value shipments.

4. RETURNS & REFUNDS:
- Returns are free if initiated within 30 days of delivery.
- Items must be in their original packaging and unused condition.
- Customers can print a pre-paid return shipping label through our online customer portal.
---
""",
)

# Define the Graph Workflow
customer_support_workflow = Workflow(
    name="customer_support_workflow",
    edges=[
        ("START", classify_query),
        (classify_query, {"shipping": shipping_faq, "unrelated": decline_to_answer}),
    ],
)

app = App(
    root_agent=customer_support_workflow,
    name="app",
)
