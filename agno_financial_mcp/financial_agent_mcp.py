
import os
import sys
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools

# Load environment variables
if not os.path.isfile(".env"):
    print("❌ ERROR: .env file not found. Please ensure it exists in the project root.")
    sys.exit(1)

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("❌ ERROR: GROQ_API_KEY is missing in your environment.")
    sys.exit(1)

# Initialize tools
finance_tools = YFinanceTools(
    stock_price=True,
    company_info=True,
    analyst_recommendations=True,
    company_news=True
)
web_tools = DuckDuckGoTools()

# Initialize Groq model—using a supported, production-ready model
llm = Groq(id="llama-3.3-70b-versatile", api_key=api_key)

agent = Agent(
    name="FinancialMCPAgent",
    model=llm,
    tools=[finance_tools, web_tools],
    markdown=True
)

if __name__ == "__main__":
    print("✅ Financial MCP Agent is ready!")
    while True:
        query = input("💬 Ask your question (or type 'exit' to quit): ")
        if query.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break
        agent.print_response(query)
