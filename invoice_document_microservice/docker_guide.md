# Master Guide: Docker Deployment & Hardware Acceleration

This document provides complete, step-by-step instructions to set up, optimize, and run the Document Extraction Microservice on any Windows computer using Docker from scratch.

---

## Step 1: Install Docker Desktop
1.  Download **Docker Desktop for Windows** from the official site: **[docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/)**
2.  Install it using the default settings, restart your computer if prompted, and open the Docker Desktop application.

---

## Step 2: Configure Hardware Acceleration (WSL2 Memory Boost)
By default, Windows limits Docker to only use 50% of your system RAM, causing local AI models to run very slowly. Follow these steps to allocate 8 GB of RAM and 4 CPU cores to Docker:

1.  Open **PowerShell** on the Windows machine.
2.  Copy and run this command to automatically generate the configuration file:
    ```powershell
    Set-Content -Path "$env:USERPROFILE\.wslconfig" -Value "[wsl2]`nmemory=8GB`nprocessors=4"
    ```
3.  Shut down the Windows subsystem to apply the new memory limits:
    ```powershell
    wsl --shutdown
    ```
4.  Open **Docker Desktop** again. It will automatically boot up with access to the full 8 GB of RAM and 4 CPU cores.

---

## Step 3: Clone the Repository
Open a Command Prompt or PowerShell and clone the project repository:
```bash
git clone https://github.com/EELiaHS-sustainability/science-tools.git
```

---

## Step 4: Run the One-Click Startup Script
1.  Open the cloned project folder in File Explorer:
    `science-tools/invoice_document_microservice/`
2.  Double-click the **`start_system.bat`** file.
3.  **What the script does automatically:**
    *   Starts the FastAPI web server container in the background.
    *   Connects to the Ollama container and automatically downloads the 1.6 GB `gemma2:2b` model.
    *   *This initial download will take 3 to 8 minutes depending on your internet connection. Subsequent launches will take less than 5 seconds.*

---

## Step 5: Execute Invoices & Document Extractions
1.  Open your web browser and navigate to: **[http://localhost:8000/docs](http://localhost:8000/docs)**
2.  Click on the **`POST /api/v1/extract`** row to expand it.
3.  Click the **"Try it out"** button.
4.  Click **"Choose File"** and upload your invoice (PDF, PNG, JPG, or TXT).
5.  Click the blue **"Execute"** button.
    *   **Cold Start (First Request)**: Will take about 1.5 to 2 minutes because the CPU has to load the 1.6 GB model into RAM for the first time.
    *   **Warm Start (Subsequent Requests)**: Will finish in **under 15 seconds** because the model stays cached in RAM (thanks to our `OLLAMA_KEEP_ALIVE=60m` setting).
