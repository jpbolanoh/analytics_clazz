# BUSINESS SCIENCE
# Pandas Data Analyst App
# -----------------------

# This app is designed to help you analyze data and create data visualizations from natural language requests.

# Imports
# !pip install git+https://github.com/business-science/ai-data-science-team.git --upgrade

from openai import OpenAI

import streamlit as st
import pandas as pd
import plotly.io as pio
import json
import os
from dotenv import load_dotenv
import re

from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import ChatOpenAI

from ai_data_science_team import (
    PandasDataAnalyst,
    DataWranglingAgent,
    DataVisualizationAgent,
)

# * APP INPUTS ----

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    # Set the API key for OpenAI
    client = OpenAI(api_key=api_key)
    
    # Test the API key (optional)
    try:
        # Example: Fetch models to validate the key
        models = client.models.list()
        # No need to show success message since this is now behind the scenes
    except Exception as e:
        st.error(f"Invalid API Key in .env file: {str(e)}")
        st.stop()
else:
    st.error("No OpenAI API Key found in .env file. Please add your API key to the .env file.")
    st.stop()

MODEL_LIST = ["gpt-4o-mini", "gpt-4o"]
TITLE = "Welcome to Galileo AI Agent!"

# ---------------------------
# Streamlit App Configuration
# ---------------------------

st.set_page_config(
    page_title=TITLE,
    page_icon="📊",
)

# Hide Streamlit menu and footer
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .stDeployButton {display:none;}
            .viewerBadge_link__qRIco {display: none !important;}
            .viewerBadge_container__1QSob {display: none !important;}
            .stToolbar {display:none;}
            .stDecoration {display:none;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
st.title(TITLE)

st.markdown("""
Welcome to Galileo AI Agent. Upload a CSV or Excel file and ask questions about the data.  
The AI agent will analyze your dataset and return interactive charts, data tables, and key insights.
""")

with st.expander("Example Questions", expanded=False):
    st.write(
        """
        ##### Bikes Data Set:
        
        -  Show the top 5 bike models by extended sales.
        -  Show the top 5 bike models by extended sales in a bar chart.
        -  Show the top 5 bike models by extended sales in a pie chart.
        -  Make a plot of extended sales by month for each bike model. Use a plotly dropdown to filter by bike model.
        -  Analyze the sales trend and provide insights on seasonality.
        """
    )

# ---------------------------
# OpenAI API Key Entry and Test
# ---------------------------

# st.sidebar.header("Enter your OpenAI API Key")

# st.session_state["OPENAI_API_KEY"] = st.sidebar.text_input(
#     "API Key",
#     type="password",
#     help="Your OpenAI API key is required for the app to function.",
# )

# Test OpenAI API Key
# if st.session_state["OPENAI_API_KEY"]:
#     # Set the API key for OpenAI
#     client = OpenAI(api_key=st.session_state["OPENAI_API_KEY"])

#     # Test the API key (optional)
#     try:
#         # Example: Fetch models to validate the key
#         models = client.models.list()
#         st.success("API Key is valid!")
#     except Exception as e:
#         st.error(f"Invalid API Key: {e}")
# else:
#     st.info("Please enter your OpenAI API Key to proceed.")
#     st.stop()


# * OpenAI Model Selection

# model_option = st.sidebar.selectbox("Choose OpenAI model", MODEL_LIST, index=0)

llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)


# ---------------------------
# File Upload and Data Preview
# ---------------------------

st.markdown("""
Upload a CSV or Excel file and ask questions about your data.  
The AI agent will analyze your dataset and return interactive charts, data tables, and key insights.
""")

uploaded_file = st.file_uploader(
    "Choose a CSV or Excel file", type=["csv", "xlsx", "xls"]
)
if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Data Preview")
    st.dataframe(df.head())
else:
    st.info("Please upload a CSV or Excel file to get started.")
    st.stop()

# ---------------------------
# Initialize Chat Message History and Storage
# ---------------------------

msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("Hola!, como puedo ayudarte?")

if "plots" not in st.session_state:
    st.session_state.plots = []

if "dataframes" not in st.session_state:
    st.session_state.dataframes = []

if "insights" not in st.session_state:
    st.session_state.insights = []
    
if "conclusions" not in st.session_state:
    st.session_state.conclusions = []





def display_chat_history():
    # Add custom CSS for message alignment
    st.markdown("""
    <style>
    .user-message {
        background-color: #262730;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
        max-width: 80%;
        margin-left: auto;
        margin-right: 10px;
        text-align: right;
    }
    .ai-message {
        background-color: #262730;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
        max-width: 80%;
        margin-left: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    for msg in msgs.messages:
        if msg.type == "human":
            # Create right-aligned user message
            col1, col2 = st.columns([2, 8])
            with col2:
                st.markdown(f'<div class="user-message">{msg.content}</div>', unsafe_allow_html=True)
        else:
            # Create left-aligned AI message
            col1, col2 = st.columns([8, 2])
            with col1:
                if "PLOT_INDEX:" in msg.content:
                    plot_index = int(msg.content.split("PLOT_INDEX:")[1])
                    st.plotly_chart(
                        st.session_state.plots[plot_index], key=f"history_plot_{plot_index}"
                    )
                elif "DATAFRAME_INDEX:" in msg.content:
                    df_index = int(msg.content.split("DATAFRAME_INDEX:")[1])
                    st.dataframe(
                        st.session_state.dataframes[df_index],
                        key=f"history_dataframe_{df_index}"
                    )
                elif "INSIGHT_INDEX:" in msg.content:
                    insight_index = int(msg.content.split("INSIGHT_INDEX:")[1])
                    st.info(st.session_state.insights[insight_index])
                elif "CONCLUSION_INDEX:" in msg.content:
                    conclusion_index = int(msg.content.split("CONCLUSION_INDEX:")[1])
                    st.success(st.session_state.conclusions[conclusion_index])
                else:
                    st.markdown(f'<div class="ai-message">{msg.content}</div>', unsafe_allow_html=True)


# Render current messages from StreamlitChatMessageHistory
display_chat_history()


# ---------------------------
# AI Agent Setup
# ---------------------------

LOG = False

pandas_data_analyst = PandasDataAnalyst(
    model=llm,
    data_wrangling_agent=DataWranglingAgent(
        model=llm,
        log=LOG,
        bypass_recommended_steps=True,
        n_samples=100,
    ),
    data_visualization_agent=DataVisualizationAgent(
        model=llm,
        n_samples=100,
        log=LOG,
    ),
)

# Function to generate insights from data
def generate_insights(data, query):
    insight_prompt = f"""
    Basado en la consulta "{query}" y los datos proporcionados, genera de 3 a 5 ideas clave que serían valiosas para el usuario. 
    Enfócate en tendencias, patrones, valores atípicos o cualquier observación significativa. Mantén cada idea concisa (1-2 oraciones).
    IMPORTANTE: Si mencionas valores monetarios, formátealos con el símbolo de dólar ($) y separadores de miles (,).
    Responde completamente en español.
    """
    
    # Convert dataframe to JSON for the prompt
    if isinstance(data, pd.DataFrame):
        data_sample = data.head(20).to_json(orient="records")
    else:
        data_sample = json.dumps(data)
    
    # Get insights from OpenAI
    insights_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Eres un analista de datos senior especializado en extraer ideas significativas de los datos. Siempre respondes en español."},
            {"role": "user", "content": f"{insight_prompt}\n\nData: {data_sample}"}
        ],
        temperature=0.3,
        max_tokens=300
    )
    
    # Extract insights
    insights_text = insights_response.choices[0].message.content
    return insights_text

# Function to generate a concise conclusion
def generate_conclusion(insights):
    conclusion_prompt = """
    Basado en los siguientes insights de datos:
    
    {insights}
    
    Genera una conclusión concisa (máximo 2 oraciones) que resuma el hallazgo más importante o la recomendación principal.
    La conclusión debe ser directa y accionable, enfocándose en lo que el usuario debería entender o hacer basado en estos datos.
    IMPORTANTE: Si mencionas valores monetarios, formátealos con el símbolo de dólar ($) y separadores de miles (,).
    """
    
    # Get conclusion from OpenAI
    conclusion_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Eres un analista de datos ejecutivo que puede sintetizar información compleja en conclusiones muy concisas. Siempre respondes en español."},
            {"role": "user", "content": conclusion_prompt.format(insights=insights)}
        ],
        temperature=0.3,
        max_tokens=75
    )
    
    # Extract conclusion
    conclusion_text = conclusion_response.choices[0].message.content
    return conclusion_text


def add_followup_message():
    followup_text = "¿Te puedo ayudar con algo más?"
    # Add to chat history
    msgs.add_ai_message(followup_text)
    # Display in the UI
    col1, col2 = st.columns([8, 2])
    with col1:
        st.markdown(f'<div class="ai-message">{followup_text}</div>', unsafe_allow_html=True)
    

# ---------------------------
# Chat Input and Agent Invocation
# ---------------------------

if question := st.chat_input("Ingresa tu pregunta aquí:", key="query_input"):
    
    with st.spinner("Pensando..."):
        col1, col2 = st.columns([2, 8])
        with col2:
            st.markdown(f'<div class="user-message">{question}</div>', unsafe_allow_html=True)
        msgs.add_user_message(question)

        try:
            pandas_data_analyst.invoke_agent(
                user_instructions=question,
                data_raw=df,
            )
            result = pandas_data_analyst.get_response()
        except Exception as e:
            st.chat_message("ai").write(
                f"An error occurred while processing your query: {str(e)}"
            )
            msgs.add_ai_message(
                f"An error occurred while processing your query: {str(e)}"
            )
            st.stop()

        routing = result.get("routing_preprocessor_decision")
        
        # Add section for insights
        st.chat_message("ai").write("🔍 Analizando tus datos...")

        if routing == "chart" and not result.get("plotly_error", False):
            # Process chart result
            plot_data = result.get("plotly_graph")
            data_wrangled = result.get("data_wrangled")
            
            if plot_data:
                # Convert dictionary to JSON string if needed
                if isinstance(plot_data, dict):
                    plot_json = json.dumps(plot_data)
                else:
                    plot_json = plot_data
                plot_obj = pio.from_json(plot_json)
                response_text = "Aquí está la visualización basada en tu consulta:"
                # Store the chart
                plot_index = len(st.session_state.plots)
                st.session_state.plots.append(plot_obj)
                msgs.add_ai_message(response_text)
                msgs.add_ai_message(f"PLOT_INDEX:{plot_index}")
                col1, col2 = st.columns([8, 2])
                with col1:
                    st.markdown(f'<div class="ai-message">{response_text}</div>', unsafe_allow_html=True)
                st.plotly_chart(plot_obj)
                
                # Generate and display insights
                insights = generate_insights(data_wrangled, question)
                insight_index = len(st.session_state.insights)
                st.session_state.insights.append(insights)
                
                # Generate conclusion
                conclusion = generate_conclusion(insights)
                conclusion_index = len(st.session_state.conclusions)
                st.session_state.conclusions.append(conclusion)
                
                # Display insights and conclusion
                st.subheader("📊 Ideas Clave")
                st.info(insights)
                
                st.subheader("🎯 Conclusión")
                st.success(conclusion)
                
                msgs.add_ai_message("📊 Ideas Clave:")
                msgs.add_ai_message(f"INSIGHT_INDEX:{insight_index}")
                msgs.add_ai_message("🎯 Conclusión:")
                msgs.add_ai_message(f"CONCLUSION_INDEX:{conclusion_index}")
                add_followup_message()
                
            else:
                st.chat_message("ai").write("The agent did not return a valid chart.")
                msgs.add_ai_message("The agent did not return a valid chart.")

        elif routing == "table":
            # Process table result
            data_wrangled = result.get("data_wrangled")
            if data_wrangled is not None:
                response_text = "Aquí está la tabla basada en tu consulta:"
                # Ensure data_wrangled is a DataFrame
                if not isinstance(data_wrangled, pd.DataFrame):
                    data_wrangled = pd.DataFrame(data_wrangled)
                df_index = len(st.session_state.dataframes)
                st.session_state.dataframes.append(data_wrangled)
                msgs.add_ai_message(response_text)
                msgs.add_ai_message(f"DATAFRAME_INDEX:{df_index}")
                col1, col2 = st.columns([8, 2])
                with col1:
                    st.markdown(f'<div class="ai-message">{response_text}</div>', unsafe_allow_html=True)
                st.dataframe(data_wrangled)
                
                # Generate and display insights
                insights = generate_insights(data_wrangled, question)
                insight_index = len(st.session_state.insights)
                st.session_state.insights.append(insights)
                
                # Generate conclusion
                conclusion = generate_conclusion(insights)
                conclusion_index = len(st.session_state.conclusions)
                st.session_state.conclusions.append(conclusion)
                
                # Display insights and conclusion
                st.subheader("📊 Ideas Clave")
                st.info(insights)
                
                st.subheader("🎯 Conclusión")
                st.success(conclusion)
                
                msgs.add_ai_message("📊 Ideas Clave:")
                msgs.add_ai_message(f"INSIGHT_INDEX:{insight_index}")
                msgs.add_ai_message("🎯 Conclusión:")
                msgs.add_ai_message(f"CONCLUSION_INDEX:{conclusion_index}")
                add_followup_message()
                
            else:
                st.chat_message("ai").write("No table data was returned by the agent.")
                msgs.add_ai_message("No table data was returned by the agent.")
        else:
            # Fallback if routing decision is unclear or if chart error occurred
            data_wrangled = result.get("data_wrangled")
            if data_wrangled is not None:
                response_text = (
                    "I apologize. There was an issue with generating the chart. "
                    "Returning the data table instead."
                )
                if not isinstance(data_wrangled, pd.DataFrame):
                    data_wrangled = pd.DataFrame(data_wrangled)
                df_index = len(st.session_state.dataframes)
                st.session_state.dataframes.append(data_wrangled)
                msgs.add_ai_message(response_text)
                msgs.add_ai_message(f"DATAFRAME_INDEX:{df_index}")
                col1, col2 = st.columns([8, 2])
                with col1:
                    st.markdown(f'<div class="ai-message">{response_text}</div>', unsafe_allow_html=True)
                st.dataframe(data_wrangled)
                
                # Generate and display insights even in fallback case
                insights = generate_insights(data_wrangled, question)
                insight_index = len(st.session_state.insights)
                st.session_state.insights.append(insights)
                
                # Generate conclusion
                conclusion = generate_conclusion(insights)
                conclusion_index = len(st.session_state.conclusions)
                st.session_state.conclusions.append(conclusion)
                
                # Display insights and conclusion
                st.subheader("📊 Ideas Clave")
                st.info(insights)
                
                st.subheader("🎯 Conclusión")
                st.success(conclusion)
                
                msgs.add_ai_message("📊 Ideas Clave:")
                msgs.add_ai_message(f"INSIGHT_INDEX:{insight_index}")
                msgs.add_ai_message("🎯 Conclusión:")
                msgs.add_ai_message(f"CONCLUSION_INDEX:{conclusion_index}")
                add_followup_message()
                
            else:
                response_text = (
                    "An error occurred while processing your query. Please try again."
                )
                msgs.add_ai_message(response_text)
                col1, col2 = st.columns([8, 2])
                with col1:
                    st.markdown(f'<div class="ai-message">{response_text}</div>', unsafe_allow_html=True)