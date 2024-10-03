import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the connection to the SQLite database
@st.cache_resource
def get_db_connection():
    conn = sqlite3.connect('logbook_attendance.db', check_same_thread=False)
    return conn

# Create tables for logbook and attendance records if they don't exist
def create_tables(conn):
    with conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS logbook (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                activity TEXT,
                date TEXT,
                description TEXT
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_name TEXT,
                roll_number TEXT,
                date TEXT,
                status TEXT
            )
        ''')

# Function to insert log entry into the logbook
def add_log_entry(conn, name, activity, date, description):
    with conn:
        conn.execute('''
            INSERT INTO logbook (name, activity, date, description)
            VALUES (?, ?, ?, ?)
        ''', (name, activity, date, description))

# Function to retrieve log entries from the logbook
def get_log_entries(conn):
    with conn:
        df = pd.read_sql_query("SELECT * FROM logbook ORDER BY date DESC", conn)
    return df

# Function to mark attendance
def mark_attendance(conn, student_name, roll_number, status):
    with conn:
        conn.execute('''
            INSERT INTO attendance (student_name, roll_number, date, status)
            VALUES (?, ?, ?, ?)
        ''', (student_name, roll_number, datetime.now().strftime("%Y-%m-%d"), status))

# Function to retrieve attendance records
def get_attendance(conn):
    with conn:
        df = pd.read_sql_query("SELECT * FROM attendance ORDER BY date DESC", conn)
    return df

# AI suggestion based on cosine similarity of descriptions
def ai_suggestions(new_description, logbook_df):
    if logbook_df.empty:
        st.warning("No logs available for AI suggestions.")
        return None
    
    vectorizer = CountVectorizer().fit_transform(logbook_df['description'].values)
    vectors = vectorizer.toarray()

    new_vector = vectorizer.transform([new_description]).toarray()
    similarity = cosine_similarity(new_vector, vectors).flatten()

    # Get the most similar log entry
    best_match_idx = similarity.argmax()
    best_match = logbook_df.iloc[best_match_idx]

    st.subheader("AI Suggested Similar Log Entry")
    st.write(f"**Name**: {best_match['name']}")
    st.write(f"**Activity**: {best_match['activity']}")
    st.write(f"**Date**: {best_match['date']}")
    st.write(f"**Description**: {best_match['description']}")
    st.write(f"**Similarity Score**: {similarity[best_match_idx]:.2f}")

# Initialize the database connection and create tables
conn = get_db_connection()
create_tables(conn)

# Streamlit App UI
st.title("Logbook and Attendance Management System")

# Page Navigation
page = st.sidebar.radio("Select a Page", ["Logbook", "Attendance Management"])

# Logbook Functionality
if page == "Logbook":
    st.header("AI Enhanced Logbook Management")

    logbook_page = st.sidebar.radio("Logbook Page", ["Add Log Entry", "View Logbook", "AI Log Suggestions"])

    if logbook_page == "Add Log Entry":
        st.subheader("Add a New Log Entry")

        # Input Form
        with st.form("log_entry_form"):
            name = st.text_input("Name")
            activity = st.selectbox("Activity", ["Lecture", "Meeting", "Task", "Event", "Other"])
            date = st.date_input("Date", datetime.today())
            description = st.text_area("Description")
            submitted = st.form_submit_button("Submit")

            # Add entry to database if the form is submitted
            if submitted:
                add_log_entry(conn, name, activity, date.strftime("%Y-%m-%d"), description)
                st.success("Log entry added successfully!")

    elif logbook_page == "View Logbook":
        st.subheader("Logbook Entries")

        # Retrieve and display log entries from the database
        logbook_df = get_log_entries(conn)
        if not logbook_df.empty:
            st.dataframe(logbook_df)

            # Filter logs by activity type
            activities = logbook_df["activity"].unique()
            selected_activities = st.multiselect("Filter by Activity", options=activities, default=activities)
            filtered_logs = logbook_df[logbook_df["activity"].isin(selected_activities)]
            
            st.write("Filtered Log Entries")
            st.dataframe(filtered_logs)
        else:
            st.warning("No log entries found.")

    elif logbook_page == "AI Log Suggestions":
        st.subheader("AI-Powered Log Insights")

        logbook_df = get_log_entries(conn)

        # Input a new description for AI suggestions
        st.subheader("Input a description to get AI-based suggestions:")
        new_description = st.text_area("Log Description")

        if st.button("Get AI Suggestion"):
            if new_description.strip() == "":
                st.warning("Please enter a description.")
            else:
                ai_suggestions(new_description, logbook_df)

# Attendance Management Functionality
elif page == "Attendance Management":
    st.header("Attendance Management")

    attendance_page = st.sidebar.radio("Attendance Page", ["Mark Attendance", "View Attendance"])

    if attendance_page == "Mark Attendance":
        st.subheader("Mark Attendance")

        # Input Form
        with st.form("attendance_form"):
            student_name = st.text_input("Student Name")
            roll_number = st.text_input("Roll Number")
            status = st.selectbox("Attendance Status", ["Present", "Absent"])
            submitted = st.form_submit_button("Submit")

            if submitted:
                mark_attendance(conn, student_name, roll_number, status)
                st.success(f"Attendance for {student_name} ({roll_number}) marked as {status}.")

    elif attendance_page == "View Attendance":
        st.subheader("Attendance Records")

        # Retrieve and display attendance records
        attendance_df = get_attendance(conn)
        if not attendance_df.empty:
            st.dataframe(attendance_df)
        else:
            st.warning("No attendance records found.")
