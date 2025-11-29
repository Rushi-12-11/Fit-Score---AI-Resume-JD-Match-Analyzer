import subprocess

def main():
    print("\n Launching FitScore (Streamlit UI)\n")
    try:
        subprocess.run(["streamlit", "run", "app/streamlit_app.py"])
    except FileNotFoundError:
        print("Streamlit not found. Install using: pip install streamlit")

if __name__ == "__main__":
    main()
