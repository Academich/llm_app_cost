import json
import streamlit as st

HOURS_IN_MONTH = round(24 * (365.25 / 12), 2)

def load_config() -> dict:
    with open("config.json", "r") as f:
        return json.load(f)


def main():
    config = load_config()

    st.set_page_config(page_title="LLM App Cost", page_icon=":shark:", layout="wide")
    st.title("LLM App Cost")

    col1, col2 = st.columns([1, 1])

    api_models = list(config["models"].keys())
    with col1:
        st.subheader("Input")
        api_model = st.selectbox("Select a model", api_models)
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

        # Suppose that a user makes a request for every shot
        # There are 18 holes in a match
        # The match lasts 4 hours
        # The total par is 72
        # Every hole requires 2 green strokes
        # The user makes 72 - (2 * 18) = 36 non-green strokes per match
        # The user makes 36 / 4 = 9 questions per hour
        requests_per_hour_per_user_average = st.slider(
            "Average number of requests per hour per user",
            min_value=0,
            max_value=100,
            value=9,
        )
        
    token_cost = (
        tokens_per_request_average
        * config["models"][api_model]["input_token_cost_per_million"] / 10**6
        + tokens_per_response_average
        * config["models"][api_model]["output_token_cost_per_million"] / 10**6
    )
    app_cost_external_api = total_active_users * requests_per_hour_per_user_average * HOURS_IN_MONTH * token_cost


    with col2:
        st.success(f"API cost per month: ${app_cost_external_api:.1f}")



if __name__ == "__main__":
    main()
