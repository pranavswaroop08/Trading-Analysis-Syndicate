import warnings
from typing import TypedDict
from langgraph.graph import StateGraph, END
import yfinance as yf
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from rich.console import Console
from rich.panel import Panel

# Suppress LangGraph deprecation warnings for a clean terminal
warnings.filterwarnings("ignore", category=UserWarning, module="langgraph")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize the local LLM connection
llm = ChatOllama(model="llama3", temperature=0.2) # Low temperature for more analytical, less creative responses

# ---------------------------------------------------------
# 1. DEFINE THE STATE
# ---------------------------------------------------------
class TradingState(TypedDict):
    ticker: str
    market_data: str
    news_headlines: str
    technical_analysis: str
    sentiment_analysis: str
    final_decision: str

# ---------------------------------------------------------
# 2. DEFINE THE AGENT NODES (Now with AI)
# ---------------------------------------------------------
def data_gatherer_node(state: TradingState):
    print(f"--- GATHERING DATA FOR {state['ticker']} ---")
    ticker = state["ticker"]
    stock = yf.Ticker(ticker)
    
    # Fetch Data
    hist = stock.history(period="10d")
    market_data = hist[['Close', 'Volume']].to_string()
    
    # Fetch News
    news = stock.news
    headlines = "\n".join([f"- {article.get('title')}" for article in news[:5]]) if news else "No news."

    return {"market_data": market_data, "news_headlines": headlines}

def quantitative_analyst_node(state: TradingState):
    print("--- QUANT ANALYST: PROCESSING TECHNICALS ---")
    
    messages = [
        SystemMessage(content="You are a ruthless quantitative analyst. Analyze the provided 10-day closing prices and volume. Identify any trends (uptrend, downtrend, consolidation). Keep your analysis strictly under 3 sentences."),
        HumanMessage(content=f"Here is the market data for {state['ticker']}:\n{state['market_data']}")
    ]
    
    response = llm.invoke(messages)
    return {"technical_analysis": response.content}

def sentiment_analyst_node(state: TradingState):
    print("--- SENTIMENT ANALYST: READING NEWS ---")
    
    messages = [
        SystemMessage(content="You are a financial sentiment analyst. Read the provided news headlines and determine if the overall market sentiment is Bullish, Bearish, or Neutral. Keep your analysis strictly under 3 sentences."),
        HumanMessage(content=f"Here are the latest headlines for {state['ticker']}:\n{state['news_headlines']}")
    ]
    
    response = llm.invoke(messages)
    return {"sentiment_analysis": response.content}

def risk_manager_node(state: TradingState):
    print("--- RISK MANAGER: EVALUATING RISK & FINALIZING TRADE ---")
    
    messages = [
        SystemMessage(content="You are a conservative Chief Risk Officer at a hedge fund. You must decide whether to execute a trade based on the technical and sentiment reports provided. Your final output MUST begin with the exact word BUY, SELL, or HOLD, followed by a 1-sentence justification."),
        HumanMessage(content=f"Ticker: {state['ticker']}\n\nTechnical Report:\n{state['technical_analysis']}\n\nSentiment Report:\n{state['sentiment_analysis']}\n\nMake your final decision.")
    ]
    
    response = llm.invoke(messages)
    return {"final_decision": response.content}

# ---------------------------------------------------------
# 3. BUILD AND COMPILE THE GRAPH
# ---------------------------------------------------------
workflow = StateGraph(TradingState)

workflow.add_node("data_gatherer", data_gatherer_node)
workflow.add_node("quant_analyst", quantitative_analyst_node)
workflow.add_node("sentiment_analyst", sentiment_analyst_node)
workflow.add_node("risk_manager", risk_manager_node)

workflow.set_entry_point("data_gatherer")
workflow.add_edge("data_gatherer", "quant_analyst")
workflow.add_edge("quant_analyst", "sentiment_analyst")
workflow.add_edge("sentiment_analyst", "risk_manager")
workflow.add_edge("risk_manager", END)

app = workflow.compile()


# ---------------------------------------------------------
# 4. EXECUTE THE SYSTEM (NOW WITH COLOR!)
# ---------------------------------------------------------
if __name__ == "__main__":
    # Initialize the Rich console
    console = Console()

    console.print("[bold cyan]--------------------------------------[/bold cyan]")
    console.print("[bold cyan]|   WELCOME TO THE TRADING SYNDICATE |[/bold cyan]")
    console.print("[bold cyan]--------------------------------------[/bold cyan]\n")
    
    # Colored input prompt
    user_ticker = console.input("[bold yellow]Enter a stock ticker to analyze (e.g., AAPL, TSLA): [/bold yellow]").strip().upper()
    
    initial_state = {
        "ticker": user_ticker, 
        "market_data": "",
        "news_headlines": "",
        "technical_analysis": "",
        "sentiment_analysis": "",
        "final_decision": ""
    }
    
    console.print(f"\n[italic dim]Starting Syndicate Run for {user_ticker}...[/italic dim]")
    result = app.invoke(initial_state)
    
    # ---------------------------------------------------------
    # FORMATTING THE FINAL REPORT INTO A BEAUTIFUL PANEL
    # ---------------------------------------------------------
    
    # Determine the color of the final decision to make it pop
    decision_text = result['final_decision'].upper()
    if "BUY" in decision_text:
        decision_color = "bold green"
    elif "SELL" in decision_text:
        decision_color = "bold red"
    else:
        decision_color = "bold yellow"

    # Construct the content for the panel
    report_content = (
        f"[bold blue]--- QUANTITATIVE ANALYSIS ---[/bold blue]\n"
        f"{result['technical_analysis']}\n\n"
        f"[bold magenta]--- SENTIMENT ANALYSIS ---[/bold magenta]\n"
        f"{result['sentiment_analysis']}\n\n"
        f"[{decision_color}]--- FINAL DECISION ---[/{decision_color}]\n"
        f"[{decision_color}]{result['final_decision']}[/{decision_color}]"
    )

    # Print the final panel
    console.print("\n")
    console.print(Panel(
        report_content, 
        title=f"[bold white]FINAL TRADING REPORT: {result['ticker']}[/bold white]", 
        border_style="cyan",
        expand=False
    ))