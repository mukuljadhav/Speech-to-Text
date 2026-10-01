# Setup & Deployment Guide (Multi-Container Docker)

This guide walks you through setting up and running the **Chef Agent API** and **Ollama** together inside Docker. This setup requires **no local installations** except Docker.

---

## 🏗️ Architecture Design

Both the FastAPI backend and Ollama run in their own Docker containers inside a private bridge network.

```text
┌────────────────────────────────────────────────────────┐
│                     DOCKER COMPOSE                     │
│                                                        │
│  ┌───────────────────────┐      ┌───────────────────┐  │
│  │    chef_agent_api     │ ───➔ │  ollama_service   │  │
│  │ (FastAPI Server :8000)│      │  (Gemma engine)   │  │
│  └───────────┬───────────┘      └─────────┬─────────┘  │
│              │                            │            │
│  ┌───────────▼───────────┐      ┌─────────▼─────────┐  │
│  │   SQLite DB Volume    │      │    Model Volume   │  │
│  │  (./data/recipes.db)  │      │ (./ollama_data/)  │  │
│  └───────────────────────┘      └───────────────────┘  │
└────────────────────────────────────────────────────────┘
```

---

## 📋 Prerequisites

Ensure the target machine has the following installed:
* **Docker & Docker Compose**:
  * [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows / macOS)
  * Docker Engine & Compose plugin (Linux)

---

## 🚀 Execution Steps

### 1. Clone the Repository
Clone the repository on the target machine and navigate to the project directory:
```bash
git clone https://github.com/EELiaHS-sustainability/science-tools.git
cd science-tools/chef_agent_api
```

### 2. Start the Containers
Run Docker Compose to build and start both the FastAPI server and the Ollama service:
```bash
# Build and start both containers in the background
docker compose up -d
```

### 3. Pull the LLM Model inside the Container (First-time only)
Since the containerized Ollama instance starts blank, you must instruct it to download the Gemma model. Run this command in your terminal:
```bash
docker exec -it ollama_service ollama pull gemma2:2b
```
*Note: The downloaded model is saved in the `./ollama_data` folder on your host machine. Rebuilding or restarting the containers will not delete it.*

---

## 🔌 Verification

* **View Logs**:
  ```bash
  docker compose logs -f
  ```
* **Verify Endpoints**:
  Open your web browser and visit **`http://localhost:8000/docs`** to access the interactive Swagger UI and execute test requests.
