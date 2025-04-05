import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Your Own Dashboard")

# File upload
uploaded_file = st.file_uploader("Choose a CSV file")
if uploaded_file is not None:
    st.success("✅ File uploaded successfully!")
    df = pd.read_csv(uploaded_file)

    # Data preview
    st.subheader("🔍 Data Preview")
    st.write(df.head())

    # Summary
    st.subheader("📊 Data Summary")
    st.write(df.describe())

    # Optional Filter
    st.subheader("🔎 Filter Data (Optional)")
    columns = df.columns.tolist()
    apply_filter = st.checkbox("Apply Filter", value=False)

    if apply_filter:
        selected_col = st.selectbox("Select a column to filter by", columns)
        unique_values = df[selected_col].unique()
        select_values = st.multiselect("Select value(s)", unique_values, default=unique_values[:1])
        filtered_df = df[df[selected_col].isin(select_values)]
    else:
        filtered_df = df

    st.write(f"📌 Showing {len(filtered_df)} rows of data")
    st.write(filtered_df)

    # Plotting
    st.subheader("📈 Plot Data")
    x_col = st.selectbox("Select x-axis column", columns)
    y_col = st.selectbox("Select y-axis column", columns)

    if st.button("Generate Plot"):
        st.line_chart(filtered_df.set_index(x_col)[y_col])

        # Additional matplotlib visualization
        fig, ax = plt.subplots()
        ax.scatter(filtered_df[x_col], filtered_df[y_col], color='blue')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{y_col} vs {x_col}")
        st.pyplot(fig)

else:
    st.info("Waiting for a CSV file upload.")
