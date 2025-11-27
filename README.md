## Deployment Link

url = https://binodkapadi-codeexplainer09.streamlit.app/

       https://binodkapadi-codeexplainer09.streamlit.app/


## CodeExplainer
CodeExplainer analyzes any pasted code, detects syntax errors, logical bugs, runtime issues, and null-pointer risks. It shows the exact line number of each error, explains how to fix it, and provides a corrected version of the code.
If no errors are found, it gives a simple, line-by-line explanation of how the code works.

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



## Setup Instructions (with Python)

1. Create and Activate a Virtual Environment
   
       python -m venv venv
       venv\Scripts\activate

3. Install Dependencies
   
       pip install -r requirements.txt

Or (If Needed)

       pip install streamlit   


5. Set Up Environment Variables
   
       GEMINI_API_KEY=your_google_gemini_api_key_here
   
7. Run the Streamlit App
   
       streamlit run mainapp.py

   By default, the app runs on:
   
        http://localhost:8501

  
11. To stop the Streamlit App
    
        ctrl + c

13. Deactivate the Virtual Environment (After Use)
    
        deactivate


## Deployment
   -Activate the virtual environment
   
        venv\Scripts\activate
   
   - Run the Streamlit App
     
         streamlit run mainapp.py

   By default, the app runs on:
   
        http://localhost:8501
        
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