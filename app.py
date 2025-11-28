# ================================
# 📁 CSV UPLOAD + GRAPH SECTION
# ================================

st.markdown("## 📁 Upload your CSV file")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:

    # Read CSV
    df = pd.read_csv(uploaded_file)
    st.success("CSV Uploaded Successfully!")
    
    st.markdown("### 🔍 Preview of CSV")
    st.dataframe(df.head())

    # Detect numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if numeric_cols:
        st.markdown("### 📊 Create Graph from CSV")

        col_choice = st.selectbox("Choose a numeric column:", numeric_cols)

        chart_type = st.radio(
            "Select graph type:",
            ["Bar Chart", "Line Chart", "Area Chart"]
        )

        chart_df = df.reset_index().rename(columns={'index': 'Row'})

        # Graph rendering
        if chart_type == "Bar Chart":
            chart = alt.Chart(chart_df).mark_bar().encode(
                x="Row:O",
                y=f"{col_choice}:Q",
                tooltip=[col_choice]
            )
        elif chart_type == "Line Chart":
            chart = alt.Chart(chart_df).mark_line(point=True).encode(
                x="Row:O",
                y=f"{col_choice}:Q",
                tooltip=[col_choice]
            )
        else:
            chart = alt.Chart(chart_df).mark_area().encode(
                x="Row:O",
                y=f"{col_choice}:Q",
                tooltip=[col_choice]
            )

        st.altair_chart(chart, use_container_width=True)

        # Stats
        st.markdown("### 📌 Column Statistics")
        st.write(df[col_choice].describe())

    else:
        st.warning("No numeric columns found for graphing.")
