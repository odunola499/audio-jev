EXAMPLE_REQUEST = {
  "model": "jev-latest",
  "state": {
    "customer_message": "I was charged twice for the same order. Please refund one of the payments.",
    "order_id": "A-104",
    "charges": [
      { "amount_usd": 49, "status": "captured" },
      { "amount_usd": 49, "status": "captured" }
    ]
  },
  "questions": {
    "team": {
      "type": "choice",
      "instructions": "Which team should handle this request?",
      "criteria": {
        "billing": "Payments, duplicate charges, refunds, and invoices",
        "shipping": "Delivery delays, tracking, and lost packages",
        "technical": "Bugs, outages, and integration problems",
        "other": "Anything outside these categories"
      }
    },
    "refund_requested": {
      "type": "noul",
      "instructions": "Is the customer requesting a refund?"
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is this request?",
      "criteria": [
        "Can wait",
        "Should be handled this week",
        "Needs attention today"
      ]
    }
  }
}