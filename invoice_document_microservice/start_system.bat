@echo off
echo ====================================================
echo   Starting Document Extraction Microservice Stack
echo ====================================================
echo.

echo Step 1: Launching Docker containers in detached mode...
docker-compose up -d

echo.
echo Step 2: Checking and downloading 'gemma2:2b' model...
echo (If the model is already downloaded, this will finish instantly)
docker exec -it invoice_document_microservice-ollama-1 ollama pull gemma2:2b

echo.
echo ====================================================
echo   [SUCCESS] Microservice is running offline!
echo.
echo   Open your browser to test: http://localhost:8000/docs
echo ====================================================
echo.
pause
