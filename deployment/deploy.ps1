<#
.SYNOPSIS
  Deploy (or tear down) the three-subset reservation model to an Azure ML
  managed online endpoint.

.DESCRIPTION
  Wraps the full sequence:
    1. (optional) rebuild the model bundle from MLflow
    2. set az CLI defaults (resource group + workspace)
    3. register the bundle as a custom model
    4. create the endpoint (skipped if it already exists)
    5. create/update the deployment and route all traffic to it
    6. smoke-test with example_input.json

  Run from anywhere; paths resolve relative to this script.
  Requires: az CLI + `ml` extension v2, and `az login` already done.

.EXAMPLE
  # Full deploy with defaults (rg_taiwei / TaiweiTestAML / DS2_v2)
  pwsh deployment/deploy.ps1

.EXAMPLE
  # Rebuild bundle first, bump model version
  pwsh deployment/deploy.ps1 -RebuildBundle -ModelVersion 2

.EXAMPLE
  # Tear everything down to stop billing
  pwsh deployment/deploy.ps1 -Teardown
#>
[CmdletBinding()]
param(
    [string]$ResourceGroup = "rg_taiwei",
    [string]$Workspace     = "TaiweiTestAML",
    [string]$EndpointName  = "resv-3split-endpoint",
    [string]$ModelName     = "resv-3split",
    [string]$ModelVersion  = "1",
    [string]$InstanceType  = "Standard_DS2_v2",
    [string]$DeploymentName = "blue",
    [switch]$RebuildBundle,
    [switch]$SkipTest,
    [switch]$Teardown
)

$ErrorActionPreference = "Stop"
$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Check()    { if ($LASTEXITCODE -ne 0) { throw "az command failed (exit $LASTEXITCODE)" } }

# --- Teardown path: delete deployment + endpoint to stop billing -------------
if ($Teardown) {
    Step "Tearing down endpoint '$EndpointName' (stops billing)"
    az ml online-endpoint delete --name $EndpointName `
        --resource-group $ResourceGroup --workspace-name $Workspace --yes
    Check
    Write-Host "Endpoint deleted. The registered model '$ModelName' is kept." -ForegroundColor Green
    return
}

# --- 0. (optional) rebuild bundle from MLflow -------------------------------
if ($RebuildBundle) {
    Step "Rebuilding model bundle from MLflow"
    Push-Location $ProjectRoot
    try { uv run python deployment/export_bundle.py; Check } finally { Pop-Location }
}

if (-not (Test-Path (Join-Path $ScriptDir "model_bundle/routing.json"))) {
    throw "model_bundle/ not found. Run with -RebuildBundle (or `uv run python deployment/export_bundle.py`) first."
}

# --- 1. az defaults ----------------------------------------------------------
Step "Setting az defaults: $ResourceGroup / $Workspace"
az configure --defaults group=$ResourceGroup workspace=$Workspace
Check

# --- 2. register model -------------------------------------------------------
Step "Registering model '$ModelName:$ModelVersion'"
az ml model create --name $ModelName --version $ModelVersion `
    --path (Join-Path $ScriptDir "model_bundle") --type custom_model
Check

# --- 3. create endpoint (idempotent) ----------------------------------------
Step "Ensuring endpoint '$EndpointName' exists"
az ml online-endpoint show --name $EndpointName 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    az ml online-endpoint create -f (Join-Path $ScriptDir "endpoint.yml")
    Check
} else {
    Write-Host "Endpoint already exists — skipping create." -ForegroundColor Yellow
}

# --- 4. create/update deployment + route traffic ----------------------------
Step "Creating deployment '$DeploymentName' (instance: $InstanceType) — this can take 10-20 min"
az ml online-deployment create -f (Join-Path $ScriptDir "deployment.yml") `
    --set model="azureml:$ModelName`:$ModelVersion" instance_type=$InstanceType `
    --all-traffic
Check

# --- 5. smoke test -----------------------------------------------------------
if (-not $SkipTest) {
    Step "Smoke test via example_input.json"
    az ml online-endpoint invoke --name $EndpointName `
        --request-file (Join-Path $ScriptDir "example_input.json")
    Check
}

$uri = az ml online-endpoint show --name $EndpointName --query scoring_uri -o tsv
Write-Host "`nDone. Scoring URI: $uri" -ForegroundColor Green
Write-Host "Remember: the deployment bills continuously. Run with -Teardown to stop." -ForegroundColor Yellow
