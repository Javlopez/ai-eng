from langchain_groq import ChatGroq
from langchain_core.tools import tool

llm = ChatGroq(model="llama-3.1-8b-instant")

@tool
def calculate_discount(price: float, discount_percentage: float) -> float:
    """
    Calculates the final price after applying a discount.

    Args:
        price (float): The original price of the item.
        discount_percentage (float): The discount percentage (e.g., 20 for 20%).

    Returns:
        float: The final price after the discount is applied.
    """

    if not( 0 <= discount_percentage <= 100 ):
        raise ValueError('discount_percentage must be between 0 and 100')

    discount_amount = price * (discount_percentage/100)
    return price - discount_amount

llm_with_tools = llm.bind_tools([calculate_discount])
hello_world = llm_with_tools.invoke("Hello World")
print("Content:", hello_world.content,'\n')

result = llm_with_tools.invoke("What is the price of an item that costs $9485 after a 25% discount?")
print("Content:", result.content)
print("--" * 10)
print(result.tool_calls)
args = result.tool_calls[0]['args']
print(calculate_discount.invoke(args))