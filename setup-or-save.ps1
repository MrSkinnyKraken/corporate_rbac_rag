# ===================================================
#    Project VENV: Setup or Save & Exit (PowerShell)
# EXECUTE LIKE THIS IN PowerShell: . .\setup-or-save.ps1
# ===================================================

$VENV_DIR = ".venv"

# --- MODO 1: "SAVE & EXIT" (Si el entorno ya esta activo) ---
if ($env:VIRTUAL_ENV) {
    Write-Host "[INFO] Se ha detectado un entorno virtual activo:" -ForegroundColor Cyan
    Write-Host "       $env:VIRTUAL_ENV"
    
    Write-Host "`n[INFO] Guardando las dependencias actuales en requirements.txt..." -ForegroundColor Cyan
    # Usamos Out-File para evitar problemas de codificación UTF-16 nativos de PowerShell
    pip freeze | Out-File -FilePath "requirements.txt" -Encoding utf8
    Write-Host "[OK] requirements.txt actualizado con exito." -ForegroundColor Green

    Write-Host "`n[INFO] Desactivando el entorno..." -ForegroundColor Cyan
    deactivate

    Write-Host "`n==================================================="
    Write-Host "    ¡Entorno guardado y proceso finalizado!" -ForegroundColor Green
    Write-Host "==================================================="
    Read-Host "`nPresiona Enter para salir..."
    exit
}

# --- MODO 2: "SETUP / START" (Si el entorno NO esta activo) ---

# Comprobar si Python esta instalado (buscamos el comando en lugar de usar %ERRORLEVEL%)
if (-not (Get-Command "python" -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] Python no esta instalado o no esta en las variables de entorno (PATH)." -ForegroundColor Red
    Write-Host "Instala Python y vuelve a ejecutar este script."
    Read-Host "`nPresiona Enter para salir..."
    exit
}

# Comprobar si el entorno virtual ya existe
if (Test-Path "$VENV_DIR\Scripts\Activate.ps1") {
    Write-Host "[INFO] El entorno virtual '$VENV_DIR' ya existe. Saltando creacion..." -ForegroundColor Cyan
} else {
    Write-Host "[INFO] Creando un nuevo entorno virtual en '$VENV_DIR'..." -ForegroundColor Cyan
    # El símbolo '&' le dice a PowerShell que ejecute el comando externo
    & python -m venv $VENV_DIR
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Hubo un fallo al crear el entorno virtual." -ForegroundColor Red
        Read-Host "`nPresiona Enter para salir..."
        exit
    }
    Write-Host "[OK] Entorno virtual creado." -ForegroundColor Green
}

Write-Host "`n[INFO] Activando el entorno virtual..." -ForegroundColor Cyan
# IMPORTANTE: El "punto y espacio" inicial es para hacer "dot-sourcing" y que el entorno no se cierre al terminar el script
. ".\$VENV_DIR\Scripts\Activate.ps1"

Write-Host "`n[INFO] Actualizando el gestor de paquetes (pip)..." -ForegroundColor Cyan
& python -m pip install --upgrade pip | Out-Null

# Comprobar dependencias e instalarlas
if (Test-Path "requirements.txt") {
    Write-Host "`n[INFO] Instalando dependencias desde requirements.txt..." -ForegroundColor Cyan
    & pip install -r requirements.txt
    Write-Host "[OK] Dependencias instaladas." -ForegroundColor Green
} else {
    Write-Host "`n[WARNING] No se encontro el archivo 'requirements.txt' en este directorio." -ForegroundColor Yellow
    Write-Host "Asegurate de crearlo para instalar las librerias del proyecto."
}

Write-Host "`n==================================================="
Write-Host "    ¡Setup completado con exito!" -ForegroundColor Green
Write-Host "==================================================="
Write-Host "NOTA: Ahora estas dentro del entorno virtual."
Write-Host "Para guardar tus dependencias y salir, vuelve a ejecutar este script."
Read-Host "`nPresiona Enter para continuar..."