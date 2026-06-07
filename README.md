# CodeExplainer
CodeExplainer analyzes any pasted code, detects syntax errors, logical bugs, runtime issues, and null-pointer risks. It shows the exact line number of each error, explains how to fix it, and provides a corrected version of the code.
If no errors are found, it gives a simple, line-by-line explanation of how the code works.

# Deployment Link

Deployment (Streamlit) = https://binodkapadi-codeexplainer09.streamlit.app

# PROJECT SETUP
pip = Python Package Installer

venv = Python Virtual Environment

### Step 1: Install Required Software

#### A) Install Python

Download and install Python:

Official Website: https://www.python.org/downloads/

During installation:

Check the option "Add Python to PATH"
Click Install Now

Verify installation [Open Command Prompt in Windows]:

     python --version

#### B) Install Visual Studio Code (Recommended)

Download and install Visual Studio Code:

Official Website: https://code.visualstudio.com

Recommended Extensions:

     Python
     Pylance
     Streamlit

### Step 2: Setup Folder Structure

Open VS Code Terminal and create a new folder:

     mkdir CodeExplainer
     cd CodeExplainer
     python -m venv venv
     venv\Scripts\activate

First of all, inside the folder create:

* .env
* requirements.txt

#### Install Dependencies

First put all required dependencies inside requirements.txt file and then run:

    pip install -r requirements.txt

Or install manually if needed:

      pip install streamlit google-generativeai python-dotenv

#### Configure Environment Variables (.env)

     GEMINI_API_KEY=your_gemini_api_key


Get Gemini API Key from: https://aistudio.google.com/app/apikey

### Step 3: Run the Streamlit Application

#### Run the application:

      streamlit run main.py

By default, the app runs on:

      http://localhost:8501

#### STOP APPLICATION

To stop the Streamlit server:

      CTRL + C

#### DEACTIVATE VIRTUAL ENVIRONMENT

After completing your work:

      deactivate



## Problem Statement
Students and developers frequently face confusing errors in their code, including syntax issues, logical bugs, and unexpected crashes. Debugging manually is time-consuming and difficult, especially for beginners who cannot identify the exact cause or line where the problem exists.


## Solution Summary
CodeExplainer is an AI-powered debugging and learning tool that deeply analyzes code to identify syntax mistakes, logical errors, runtime failures, and null-pointer risks. It reports the exact error lines, explains the cause clearly, and provides a corrected version of the full code.
If the code has no issues, CodeExplainer automatically switches to explanation mode and delivers a clean, line-by-line breakdown for easy understanding.

## Tech Stack

    - Backend: Python, Streamlit
    - Frontend: Streamlit Components, Custom CSS
    - AI / LLM Models: Google Gemini 2.0 Flash (google-generativeai SDK)
    - Deployment / Hosting: Streamlit Cloud
    - Version Control: Git and GitHub

## Project Structure
CODEEXPLAINER

    - main.py                       # Main Streamlit application
    - style.css                     # Custom UI styling (Dark Mode)
    - .env                          # Environment variables (contains GEMINI_API_KEY)
    - requirements.txt              # Project dependencies
    - README.md                     # Project documentation
    - .gitignore                    # For hiding api key ( or other sensitive information)
    -  venv/                        # Virtual environment directory 

## Features
- Error Detection => Identifies syntax errors, logical bugs, runtime issues, and null-pointer risks in any programming language code.
- Exact Line Highlighting => Shows the precise line numbers where errors occur.
- Clear Fix Suggestions => Explains the cause of each error and provides step-by-step solutions.
- Auto-Corrected Code => Generates a fully corrected version of the entire code at the end.
- Code Understanding Mode => If no errors exist, it gives a clean, beginner-friendly, line-by-line explanation of the code.
- Multi-Language Support => Works with C++, Java, Python, JavaScript, and more.
- Readable Output => Provides organized, easy-to-understand explanations and corrections.


## Technical Architecture
The CodeExplainer architecture uses a Streamlit frontend where users paste code. This code is sent to a Python backend powered by Google GenAI, which forwards the input to the Google Gemini API for analysis. The API detects syntax, logic, and runtime issues or generates a line-by-line explanation. The backend then formats the results—error reports, fixes, or explanations—and returns clean, readable output to the frontend for display.

     ASCII Architecture Diagram:
     
           Frontend (Streamlit UI)
                   ↓  User pastes code
           Backend (Python + Google GenAI)
                   ↓
           Google Gemini API
                   ↓
           Code is analyzed for syntax, logic & runtime issues
                   ↓
           Backend maps errors, fixes code, or explains line-by-line
                   ↓
           Frontend displays errors, corrections, or explanations



## References

      - Google Gemini API Documentation 
      - Streamlit Official Documentation
      - Python Standard Library 
      - Code Parsing & Static Analysis Concepts 


## Acknowledgements
- Developed by Binod Kapadi
- Special thanks to Google Gemini for enabling AI-powered code understanding, explanation, and optimization.