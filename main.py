import os
from typing import Annotated, List, Tuple, TypedDict, Union
from pydantic import BaseModel, Field
import requests
from langchain.tools import StructuredTool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from flask import Flask, request, jsonify
from flask_cors import CORS
import urllib3
import warnings
from dotenv import load_dotenv
load_dotenv()

# Ignore SSL warnings
warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)
urllib3.util.ssl_.DEFAULT_CIPHERS = 'ALL:@SECLEVEL=1'

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Set your API keys
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# Currency Conversion Tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> float:
    """Convert currency amount from one currency to another"""
    url = f"https://api.exchangerate-api.com/v4/latest/{from_currency}"
    response = requests.get(url, verify=False)
    data = response.json()
    rate = data['rates'][to_currency]
    converted_amount = amount * rate
    return round(converted_amount, 2)

class CurrencyConversionInput(BaseModel):
    amount: float = Field(..., description="The amount of money to convert")
    from_currency: str = Field(..., description="The currency to convert from (e.g., USD)")
    to_currency: str = Field(..., description="The currency to convert to (e.g., JPY)")

# Initialize tools and LLM
currency_conversion_tool = StructuredTool.from_function(
    name="currency_converter",
    description="Convert an amount from one currency to another",
    func=convert_currency,
    args_schema=CurrencyConversionInput
)

tavily_search = TavilySearchResults(max_results=2)

llm = ChatGroq(
    temperature=0.7,
    model_name="llama3-8b-8192",
    max_tokens=1024
)

async def create_travel_plan(query: str) -> List[str]:
    """Create a simple travel plan"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Create a numbered list of 2-3 specific steps to plan this trip. Focus on budget and practical arrangements."),
        ("human", "{query}")
    ])
    
    messages = prompt.format_messages(query=query)
    response = await llm.ainvoke(messages)
    print(response.content)
    return [step.strip() for step in response.content.split('\n') if step.strip()]

async def execute_step(step: str) -> str:
    """Execute a single step of the plan"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Provide specific information for this travel planning step.
        Include costs, practical details, and recommendations where appropriate.
        If mentioning prices, include both USD and JPY amounts."""),
        ("human", "{step}")
    ])
    
    messages = prompt.format_messages(step=step)
    response = await llm.ainvoke(messages)
    return response.content

async def run_travel_agent(plan: str):
    """Run simplified travel agent"""
    try:
        # Execute each step
        final_results = []
        for step in plan:
            print(f"\nExecuting step: {step}")
            result = await execute_step(step)
            print(f"Step result: {result}")
            final_results.append({"step": step, "details": result})
        
        return {
            "status": "success",
            "plan": final_results
        }
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }


# Flask route to handle travel planning requests
@app.route('/plan-trip', methods=['POST'])
async def plan_trip():
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                "status": "error",
                "message": "Missing 'query' in request body"
            }), 400
            
        query = data['query']
        
        # First, create and send the initial plan
        initial_plan = await create_travel_plan(query)
        print("\nTravel Plan:")
        for i, step in enumerate(initial_plan, 1):
            print(f"{i}. {step}")
        
        # Send initial plan to client
        initial_response = jsonify({
            "type": "plan",
            "content": initial_plan
        })
        initial_response.headers['X-Content-Type'] = 'plan'
        
        # Then execute the plan
        print("\nExecuting plan...\n")
        results = []
        for step in initial_plan:
            print(f"\nProcessing: {step}")
            result = await execute_step(step)
            results.append((step, result))
            print(f"Result: {result}\n")
            print("-" * 50)
        
        # Format final response
        print("\nFinal Travel Plan Summary:")
        final_details = []
        for step, result in results:
            print(f"\n## {step}")
            print(result)
            final_details.append(result)
            
        # Send final result to client
        return jsonify({
            "type": "final",
            "content": final_details
        })
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({
            "type": "error",
            "message": str(e)
        }), 500


if __name__ == "__main__":
    # Run the Flask app
    app.run(debug=True, port=5000)