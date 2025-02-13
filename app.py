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
        st.subheader("General settings")

        total_active_users = st.slider(
            "Total active users", min_value=0, max_value=5000, value=200
        )
        tokens_per_request_average = st.slider(
            "Average number of tokens per request",
            min_value=0,
            max_value=10000,
            value=2000,
        )
        tokens_per_response_average = st.slider(
            "Average number of tokens per response",
            min_value=0,
            max_value=200,
            value=90,
        )

        requests_per_hour_per_user_average = st.slider(
            "Average number of requests per hour per user",
            min_value=0,
            max_value=100,
            value=9,
        )

    with col2:
        st.subheader("Monthly cost when using an API")
        api_model = st.selectbox("Select a model", api_models)
        token_cost = (
            tokens_per_request_average
            * config["models"][api_model]["input_token_cost_per_million"]
            / 10**6
            + tokens_per_response_average
            * config["models"][api_model]["output_token_cost_per_million"]
            / 10**6
        )
        app_cost_external_api = (
            total_active_users
            * requests_per_hour_per_user_average
            * HOURS_IN_MONTH
            * token_cost
        )
        st.success(f"API cost per month: ${app_cost_external_api:.1f}")
        st.subheader("Monthly cost when using a cloud GPU")
        gpu_number = st.slider("Number of GPUs", min_value=0, max_value=10, value=1)
        gpu_cost_per_hour = st.slider("Cost per hour", min_value=0.0, max_value=5.0, value=0.99)
        gpu_cost_per_month = gpu_number * gpu_cost_per_hour * HOURS_IN_MONTH
        st.success(f"Cloud GPU cost per month: ${gpu_cost_per_month:.1f}")


if __name__ == "__main__":
    main()
