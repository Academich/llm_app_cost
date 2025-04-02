import json
import math
import streamlit as st
import numpy as np
import pandas as pd

HOURS_IN_MONTH = round(24 * (365.25 / 12), 2)


def load_config() -> dict:
    with open("config.json", "r") as f:
        return json.load(f)


def main():
    config = load_config()

    def get_GPU_throughput(gpu_name: str, llm_parameters: int) -> float:
        """
        Get the throughput of a GPU model in tokens per second.
        """
        efficiency = 0.5 # Just a guess
        FLOPS_half_precision = config["GPU"][gpu_name]["TFLOPS_half_precision"] * 10**12
        FLOPS_per_token = 6 * llm_parameters
        return FLOPS_half_precision * efficiency / FLOPS_per_token

    # Function to calculate API cost
    def calculate_api_cost(num_users, api_model, tokens_req, tokens_resp, req_per_hour):
        token_cost = (
            tokens_req * config["models"][api_model]["input_token_cost_per_million"] / 10**6
            + tokens_resp * config["models"][api_model]["output_token_cost_per_million"] / 10**6
        )
        return num_users * req_per_hour * HOURS_IN_MONTH * token_cost
    
    # Function to calculate GPU cost
    def calculate_gpu_cost(num_users, gpu_model, model_size_b_params, req_per_hour, tokens_req, tokens_resp):
        tokens_per_second = num_users * req_per_hour * (tokens_resp + tokens_req) / 3600
        throughput = get_GPU_throughput(gpu_model, model_size_b_params * 10**9)
        required_gpus = max(1, math.ceil(tokens_per_second / throughput))
        return required_gpus * config["GPU"][gpu_model]["average_dollars_per_hour_on_aws"] * HOURS_IN_MONTH


    st.set_page_config(page_title="LLM App Cost", page_icon=":shark:", layout="wide")
    st.title("LLM App Cost")

    col1, col2 = st.columns([1, 1])

    api_models = list(config["models"].keys())
    with col1:
        st.subheader("General settings")

        # Total active users
        total_active_users_col1, total_active_users_col2 = st.columns([3, 1])
        with total_active_users_col1:
            total_active_users = st.slider(
                "Total active users", min_value=0, max_value=5000, value=200, key="total_active_users_slider"
            )
        with total_active_users_col2:
            total_active_users_input = st.number_input(
                "Exact value", min_value=0, max_value=5000, value=total_active_users, key="total_active_users_input"
            )
            total_active_users = total_active_users_input
        
        # Tokens per request
        tokens_req_col1, tokens_req_col2 = st.columns([3, 1])
        with tokens_req_col1:
            tokens_per_request_average = st.slider(
                "Average number of tokens per request",
                min_value=0,
                max_value=10000,
                value=2000,
                key="tokens_per_request_slider"
            )
        with tokens_req_col2:
            tokens_per_request_input = st.number_input(
                "Exact value", min_value=0, max_value=10000, value=tokens_per_request_average, key="tokens_per_request_input"
            )
            tokens_per_request_average = tokens_per_request_input
        
        # Tokens per response
        tokens_resp_col1, tokens_resp_col2 = st.columns([3, 1])
        with tokens_resp_col1:
            tokens_per_response_average = st.slider(
                "Average number of tokens per response",
                min_value=0,
                max_value=200,
                value=90,
                key="tokens_per_response_slider"
            )
        with tokens_resp_col2:
            tokens_per_response_input = st.number_input(
                "Exact value", min_value=0, max_value=200, value=tokens_per_response_average, key="tokens_per_response_input"
            )
            tokens_per_response_average = tokens_per_response_input

        # Requests per hour
        req_hour_col1, req_hour_col2 = st.columns([3, 1])
        with req_hour_col1:
            requests_per_hour_per_user_average = st.slider(
                "Average number of requests per hour per user",
                min_value=0,
                max_value=100,
                value=9,
                key="requests_per_hour_slider"
            )
        with req_hour_col2:
            requests_per_hour_input = st.number_input(
                "Exact value", min_value=0, max_value=100, value=requests_per_hour_per_user_average, key="requests_per_hour_input"
            )
            requests_per_hour_per_user_average = requests_per_hour_input
        # Tokens per second
        required_tokens_per_second = total_active_users * requests_per_hour_per_user_average * (tokens_per_response_average + tokens_per_request_average) / 3600
        st.success(f"Required tokens per second: {required_tokens_per_second:.1f}")

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
        
        # Model size and GPU memory considerations
        model_size_col1, model_size_col2 = st.columns([3, 1])
        with model_size_col1:
            model_size_billion_parameters = st.slider("Model size (billion parameters)", min_value=1, max_value=500, value=7, key="model_size_slider")
        with model_size_col2:
            model_size_billion_parameters_input = st.number_input("Exact value", min_value=1, max_value=500, value=model_size_billion_parameters, key="model_size_input")
            model_size_billion_parameters = model_size_billion_parameters_input
            
        # GPU model selection
        gpu_model = st.selectbox(
            "Select GPU model",
            options=["T4", "L40", "V100", "A100", "H100"],
            key="gpu_model_select"
        )
        gpu_cost_per_hour = config["GPU"][gpu_model]["average_dollars_per_hour_on_aws"]
        
        # Set GPU memory based on selected model
        gpu_memory_gb = config["GPU"][gpu_model]["memory_gb"]
        st.info(f"GPU Memory: {gpu_memory_gb} GB")
        
        # Performance metrics
        GPU_throughput = get_GPU_throughput(gpu_model, model_size_billion_parameters * 10**9)
        st.success(f"GPU throughput: {GPU_throughput:.1f} tokens per second")

        # Calculate required GPUs based on throughput
        gpus_for_throughput = math.ceil(required_tokens_per_second / GPU_throughput)
        
        # TODO Calculate GPUs needed based on model size
        
        required_gpus = gpus_for_throughput
        
        st.success(f"Required GPUs: {required_gpus}")
        
        # Cost calculations
        gpu_cost_col1, gpu_cost_col2 = st.columns([3, 1])
        with gpu_cost_col1:
            gpu_cost_per_hour = st.slider(
                "GPU cost per hour ($)", min_value=0.0, max_value=10.0, value=gpu_cost_per_hour, step=0.01, key="gpu_cost_slider"
            )
        with gpu_cost_col2:
            gpu_cost_input = st.number_input("Exact value", min_value=0.0, max_value=10.0, value=gpu_cost_per_hour, step=0.01, key="gpu_cost_input")
            gpu_cost_per_hour = gpu_cost_input
        
        # Calculate total GPU cost
        gpu_cost_per_month = required_gpus * gpu_cost_per_hour * HOURS_IN_MONTH
        
        # Display cost
        st.success(f"Cloud GPU cost per month: ${gpu_cost_per_month:.2f}")

    # Add a new section for the graph
    st.header("Cost Scaling with Number of Users")
    
    # Create user range for x-axis
    user_range = np.linspace(1, 10000, 50).astype(int)
    
    # Calculate costs for each user count
    api_costs = [calculate_api_cost(users, api_model, tokens_per_request_average, 
                                   tokens_per_response_average, requests_per_hour_per_user_average) 
                for users in user_range]
    
    gpu_costs = [calculate_gpu_cost(users, gpu_model, model_size_billion_parameters,
                                   requests_per_hour_per_user_average, tokens_per_request_average,
                                   tokens_per_response_average)
                for users in user_range]
    
    # Create a DataFrame for Streamlit charting
    chart_data = pd.DataFrame({
        'Users': user_range,
        f'API Cost ({api_model})': api_costs,
        f'GPU Cost ({gpu_model})': gpu_costs
    })
    
    # Set the index to the user count for proper x-axis
    chart_data = chart_data.set_index('Users')
    
    # Find intersection point(s) if they exist
    intersection_points = []
    for i in range(1, len(user_range)):
        if (api_costs[i-1] <= gpu_costs[i-1] and api_costs[i] >= gpu_costs[i]) or \
           (api_costs[i-1] >= gpu_costs[i-1] and api_costs[i] <= gpu_costs[i]):
            # Simple linear interpolation to find intersection
            x_intersect = user_range[i-1] + (user_range[i] - user_range[i-1]) * \
                         (gpu_costs[i-1] - api_costs[i-1]) / \
                         ((api_costs[i] - api_costs[i-1]) - (gpu_costs[i] - gpu_costs[i-1]))
            y_intersect = api_costs[i-1] + (api_costs[i] - api_costs[i-1]) * \
                         (x_intersect - user_range[i-1]) / (user_range[i] - user_range[i-1])
            intersection_points.append((int(x_intersect), y_intersect))
    
    # Display the chart
    st.line_chart(chart_data, color=['#FF0000', '#1351a8'], x_label="Number of Users", y_label="Cost per Month ($)")
    
    # Add information about current user count
    st.info(f"Current user count: {total_active_users}")
    
    # Add information about break-even points
    if intersection_points:
        for x, y in intersection_points:
            st.info(f"Break-even point: {x} users at ${y:.2f}/month")
    
    # Add cost-effective recommendation
    if total_active_users > 0:
        current_api_cost = calculate_api_cost(total_active_users, api_model, tokens_per_request_average, 
                                            tokens_per_response_average, requests_per_hour_per_user_average)
        current_gpu_cost = calculate_gpu_cost(total_active_users, gpu_model, model_size_billion_parameters,
                                            requests_per_hour_per_user_average, tokens_per_request_average,
                                            tokens_per_response_average)
        
        if current_api_cost < current_gpu_cost:
            st.success(f"For {total_active_users} users, API ({api_model}) is more cost-effective: \${current_api_cost:.2f} vs \${current_gpu_cost:.2f} per month")
        else:
            st.success(f"For {total_active_users} users, GPU ({gpu_model}) is more cost-effective: \${current_gpu_cost:.2f} vs \${current_api_cost:.2f} per month")
    
    # Add explanation text
    st.markdown("""
    ### Understanding the Graph
    
    This graph shows how costs scale with the number of users:
    
    - **Red line**: API cost using the selected model
    - **Blue line**: GPU cost using the selected GPU model
    
    The more cost-effective option depends on your user count. API solutions typically have linear scaling, 
    while GPU solutions have step-wise scaling as you add more GPUs to handle increased load.
    """)


if __name__ == "__main__":
    main()
