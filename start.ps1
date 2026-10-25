# ==============================================================================
# Production RAG Full-Stack One-Click Launcher (Windows PowerShell)
# ==============================================================================

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Starting Production RAG Full-Stack Ecosystem" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Verify Virtual Environment
if (!(Test-Path ".\venv\Scripts\python.exe")) {
    Write-Host "[ERROR] Virtual environment not found at .\venv" -ForegroundColor Red
    Write-Host "Please run: python -m venv venv; .\venv\Scripts\pip install -r requirements.txt"
    exit 1
}

# 2. Check if documents are ingested
if (!(Test-Path ".\data\processed\chunks.json")) {
    Write-Host "[INFO] Ingesting documents into Qdrant..." -ForegroundColor Yellow
    .\venv\Scripts\python.exe -m ingest.ingest
}

# 3. Start Backend API in background job
Write-Host "[1/2] Starting FastAPI Backend on http://localhost:8000..." -ForegroundColor Cyan
$BackendJob = Start-Job -ScriptBlock {
    Set-Location $using:PWD
    .\venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
}

# Wait for backend readiness
$BackendReady = $false
for ($i = 0; $i -lt 15; $i++) {
    Start-Sleep -Seconds 1
    try {
        $res = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method Get -TimeoutSec 1
        if ($res.status -eq "ok") {
            $BackendReady = $true
            break
        }
    } catch {}
}

if ($BackendReady) {
    Write-Host "[SUCCESS] FastAPI Backend is live at http://localhost:8000" -ForegroundColor Green
    Write-Host "  * API Docs: http://localhost:8000/docs" -ForegroundColor Gray
    Write-Host "  * Health:   http://localhost:8000/health" -ForegroundColor Gray
    Write-Host "  * Metrics:  http://localhost:8000/metrics" -ForegroundColor Gray
} else {
    Write-Host "[WARNING] Backend is taking longer than usual to start." -ForegroundColor Yellow
}

# 4. Start Next.js Frontend
Write-Host "`n[2/2] Starting Next.js Web Frontend on http://localhost:3000..." -ForegroundColor Cyan
Set-Location .\frontend
Start-Process "http://localhost:3000"
npm run dev
