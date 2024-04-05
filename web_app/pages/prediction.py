import streamlit as st
import pandas as pd

# 初始化一个空DataFrame，用于收集用户输入的数据
data_columns = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'duration', 'campaign', 'poutcome']
user_data = pd.DataFrame(columns=data_columns)

# 判断是否已经收集了数据
if 'finished_collecting' not in st.session_state:
    st.session_state.finished_collecting = False

# 定义一个函数用于保存当前页面的数据
def save_data():
    # 这里可以添加实际的数据保存逻辑
    st.session_state.finished_collecting = True

if not st.session_state.finished_collecting:
    for column in data_columns:
        # 对于每一列，创建一个与其对应的输入字段。这里以文本输入为例，可根据数据类型调整
        user_input = st.text_input(f"Please enter your {column}:", key=column)
        user_data.at[0, column] = user_input
    
    if st.button("Submit"):
        save_data()
else:
    st.write("Finished")

# 注意：运行此Streamlit应用时，需要将每个输入字段按需调整为适当的数据类型（例如，下拉菜单、单选按钮等），以便更好地收集用户数据。