import pandas as pd
import streamlit as st

st.set_page_config("std_form", layout="centered")
st.title("Student Information Form")
st.write("Please fill in the details below:")
st.write("---")

with st.form("student_Marks_form"):
    name = st.text_input("Enter Your Name")
    Html_marks = st.number_input("Enter Your HTML Marks Obtained(out of 100)", min_value=0, max_value=100, step=1)
    Css_marks = st.number_input("Enter Your CSS Marks Obtained(out of 100)", min_value=0, max_value=100, step=1)
    python_marks = st.number_input("Enter Your Python Marks Obtained(out of 100)", min_value=0, max_value=100, step=1)
    Java_marks = st.number_input("Enter Your Java Marks Obtained(out of 100)", min_value=0, max_value=100, step=1)
    ReactJS_marks = st.number_input("Enter Your ReactJS Marks Obtained(out of 100)", min_value=0, max_value=100, step=1)
    submitted = st.form_submit_button("Submit")
    
def std_percentage(marks):
    return round(sum(marks)/500*100, 2)

def std_grade(percentage):
    if percentage >= 90:
        return "A+"
    elif percentage >= 80:
        return "A"
    elif percentage >= 70:
        return "B+"
    elif percentage >= 60:
        return "B"
    elif percentage >= 50:
        return "C"
    else:
        return "F"
    
if submitted:
    marks = [Html_marks, Css_marks, python_marks, Java_marks, ReactJS_marks]
    percentage = std_percentage(marks)
    grade = std_grade(percentage)
    
    st.write("### Student Details:")
    st.write(f"**Name:** {name}")
    st.write(f"**Percentage:** {percentage}%")
    st.write(f"**Grade:** {grade}")
    
df = pd.DataFrame({
    "Subjects": ["HTML", "CSS", "Python", "Java", "ReactJS"],
    "Marks Obtained": [Html_marks, Css_marks, python_marks, Java_marks, ReactJS_marks]
})

st.table(df)