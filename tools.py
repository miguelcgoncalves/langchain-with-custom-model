from langchain_core.tools import tool

@tool
def lookup_policy(query: str) -> str:
    """Consult the company policies to check whether certain options are permitted."""
    return """
**Order Cancellation Policy**

Thank you for choosing our products and services. Please note that all sales are final, and once an order has been placed, it cannot be canceled under any circumstances.

This no-cancellation policy ensures that we are able to prepare, process, and deliver your order as quickly and efficiently as possible. Due to the nature of our operations and/or customized offerings, we are unable to accommodate cancellation requests. We kindly ask that you review your order carefully before completing your purchase to ensure all details are correct.

If you have any questions or concerns about your order, please contact our customer service team. We are committed to assisting you to the best of our ability within the scope of our policy.

Thank you for your understanding and cooperation.
"""
