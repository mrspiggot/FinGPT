import streamlit as st
import pandas as pd
from PIL import Image

import yfinance as yf
from yahooquery import Ticker
from datetime import datetime, timedelta
from edgar import Company, TXTML

from dotenv import load_dotenv
import os

from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter, PythonCodeTextSplitter


openai_api_key = os.getenv("OPENAI_API_KEY")
load_dotenv()

def format_large_number(num):
    if abs(num) >= 1_000_000_000_000:
        return f"${num / 1_000_000_000_000:.2f}T"
    elif abs(num) >= 1_000_000_000:
        return f"${num / 1_000_000_000:.2f}B"
    elif abs(num) >= 1_000_000:
        return f"${num / 1_000_000:.2f}M"
    else:
        return str(num)

# Create a dictionary for stocks
stocks = {
    "Apple - 'AAPL'": {"name": "APPLE INC", "symbol": "AAPL", "cik": "0000320193"},
    "Alphabet - 'GOOG'": {"name": "Alphabet Inc.", "symbol": "GOOG", "cik": "0001652044"},
    "Facebook - 'META'": {"name": "META PLATFORMS INC", "symbol": "META", "cik": "0001326801"},
    "Amazon - 'AMZN'": {"name": "AMAZON COM INC", "symbol": "AMZN", "cik": "0001018724"},
    "Netflix - 'NFLX'": {"name": "NETFLIX INC", "symbol": "NFLX", "cik": "0001065280"},
    "Microsoft - 'MSFT'": {"name": "MICROSOFT CORP", "symbol": "MSFT", "cik": "0000789019"},
    "Tesla - 'TSLA'": {"name": "TESLA INC", "symbol": "TSLA", "cik": "0001318605"},
}
def get_recommendation(stock_cik, question):

    company = Company(stock_cik["name"], stock_cik["cik"])
    doc = company.get_10K()
    text = TXTML.parse_full_10K(doc)

    llm = OpenAI(temperature=0.15, openai_api_key=openai_api_key)

    lts = int(len(text) / 3)
    lte = int(lts * 2)

    text_splitter = PythonCodeTextSplitter(chunk_size=3000, chunk_overlap=300)
    docs = text_splitter.create_documents([text[lts:lte]])

    # Get your embeddings engine ready
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Embed your documents and combine with the raw text in a pseudo db. Note: This will make an API call to OpenAI
    docsearch = FAISS.from_documents(docs, embeddings)

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())
    query = question
    analysis = qa.run(query)

    return analysis.translate(str.maketrans("", "", "_*"))


st.set_page_config(page_title="Stock Information", layout="wide", initial_sidebar_state="collapsed", page_icon="Color Logo.png")
col1, col2 = st.columns((1, 3))
icon = Image.open("Colour Logo.png")
col1.image(icon, width=100)
selected_stock = col1.selectbox("Select a stock", options=list(stocks.keys()), index=0)

# Get stock data from yfinance
ticker = yf.Ticker(stocks[selected_stock]["symbol"])

# Calculate the date range for the last 30 days
end_date = datetime.now()
start_date = end_date - timedelta(days=360)

# Get the closing prices for the selected stock in the last 30 days
data = ticker.history(start=start_date, end=end_date)
closing_prices = data["Close"]

# Plot the line chart in the first column
col1.line_chart(closing_prices, use_container_width=True)

# Get the company long description
long_description = ticker.info["longBusinessSummary"]

# Display the long description in a text box in the second column
col2.title("Company Overview")
col2.write(long_description)

# Use yahooquery to get earnings and revenue
ticker_yq = Ticker(stocks[selected_stock]["symbol"])
earnings = ticker_yq.earnings

financials_data = earnings[stocks[selected_stock]["symbol"]]['financialsChart']['yearly']


df_financials = pd.DataFrame(financials_data)
df_financials['revenue'] = df_financials['revenue']
df_financials['earnings'] = df_financials['earnings']
df_financials = df_financials.rename(columns={'earnings': 'yearly earnings', 'revenue': 'yearly revenue'})

numeric_cols = ['yearly earnings', 'yearly revenue']
df_financials[numeric_cols] = df_financials[numeric_cols].applymap(format_large_number)
df_financials['date'] = df_financials['date'].astype(str)
df_financials.set_index('date', inplace=True)

# Display earnings and revenue in the first column
col1.write(df_financials)

summary_detail = ticker_yq.summary_detail[stocks[selected_stock]["symbol"]]

obj = yf.Ticker(stocks[selected_stock]["symbol"])

pe_ratio = '{0:.2f}'.format(summary_detail["trailingPE"])
price_to_sales = summary_detail["fiftyTwoWeekLow"]
target_price = summary_detail["fiftyTwoWeekHigh"]
market_cap = summary_detail["marketCap"]
ebitda = ticker.info["ebitda"]
tar = ticker.info["targetHighPrice"]
rec = ticker.info["recommendationKey"].upper()

# Format large numbers
market_cap = format_large_number(market_cap)
ebitda = format_large_number(ebitda)

# Create a dictionary for additional stock data
additional_data = {
    "P/E Ratio": pe_ratio,
    "52 Week Low": price_to_sales,
    "52 Week High": target_price,
    "Market Capitalisation": market_cap,
    "EBITDA": ebitda,
    "Price Target": tar,
    "Recommendation": rec
}

# Display additional stock data in the first column
for key, value in additional_data.items():
    col1.write(f"{key}: {value}")

st.title("Lucidate Research Demo. Powered by LangChain 🦜🔗 ")
col2.title("Opportunities for investors")
print(f"**********\nstocks[selected_stock]\n*************\n{stocks[selected_stock]}\n\n**********\n")
col2.write(get_recommendation(stocks[selected_stock], "What are this firm's key products and services?"))
col2.write(get_recommendation(stocks[selected_stock], "What are the new products and growth opportunities for this firm. What are its unique strengths?"))
col2.write(get_recommendation(stocks[selected_stock], "Who are this firms key competitors? What are the principal threats?"))


