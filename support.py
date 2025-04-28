import subprocess

# Define components and their ports
components = {
    "predictor.py": 8501,
    "stock_glance.py": 8502,
    "stock_symbol.py": 8503,
    "news.py": 8504,
}

# Launch each component
for script, port in components.items():
    subprocess.Popen(["streamlit", "run", script, "--server.port", str(port)])
subprocess.Popen(["python", "converter.py", "--server.port", "5000"])

print("✅ All components launched. Access them via their respective ports.")
