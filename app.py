import json
import streamlit as st
from config import (
    GPT_35_TURBO_PROMPT_COST,
    GPT_35_TURBO_COMPLETION_COST,
    GPT_4_PROMPT_COST,
    GPT_4_COMPLETION_COST,
)

HOURS_IN_MONTH = round(24 * (365.25 / 12), 2)

def load_config() -> dict:
    with open("config.json", "r") as f:
        return json.load(f)


def main():
    config = load_config()

    st.set_page_config(page_title="LLM App Cost", page_icon=":shark:", layout="wide")
    st.title("LLM App Cost")

    col1, col2 = st.columns([1, 1])

    api_models = list(config["API_token_cost"].keys())
    with col1:
        st.subheader("Input")
        option = st.selectbox("Select a model", api_models)
        total_active_users = st.slider("Total active users", min_value=0, max_value=5000, value=200)
        tokens_per_request_average = st.slider(
            "Average number of tokens per request",
            min_value=0,
            max_value=10000,
            value=100,
        )
        tokens_per_response_average = st.slider(
            "Average number of tokens per response",
            min_value=0,
            max_value=10000,
            value=100,
        )
        requests_per_hour_per_user_average = st.slider(
            "Average number of requests per hour per user",
            min_value=0,
            max_value=100,
            value=1,
        )
        
    token_cost = (
        tokens_per_request_average
        * config["API_token_cost"][option]["input_token_cost_per_million"] / 10**6
        + tokens_per_response_average
        * config["API_token_cost"][option]["output_token_cost_per_million"] / 10**6
    )
    app_cost_external_api = total_active_users * requests_per_hour_per_user_average * HOURS_IN_MONTH * token_cost


    with col2:
        st.success(f"API cost per month: ${app_cost_external_api:.1f}")



if __name__ == "__main__":
    main()
