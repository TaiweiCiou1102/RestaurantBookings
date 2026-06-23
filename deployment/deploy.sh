#!/usr/bin/env bash
# Deploy (or tear down) the three-subset reservation model to an Azure ML
# managed online endpoint. Portable counterpart of deploy.ps1.
#
# Sequence: (opt) rebuild bundle -> set defaults -> register model ->
#           create endpoint (idempotent) -> create/update deployment -> smoke test
#
# Requires: az CLI + `ml` extension v2, and `az login` already done.
#
# Usage:
#   deployment/deploy.sh                         # full deploy with defaults
#   REBUILD=1 MODEL_VERSION=2 deployment/deploy.sh
#   TEARDOWN=1 deployment/deploy.sh              # delete endpoint to stop billing
set -euo pipefail

RESOURCE_GROUP="${RESOURCE_GROUP:-rg_taiwei}"
WORKSPACE="${WORKSPACE:-TaiweiTestAML}"
ENDPOINT_NAME="${ENDPOINT_NAME:-resv-3split-endpoint}"
MODEL_NAME="${MODEL_NAME:-resv-3split}"
MODEL_VERSION="${MODEL_VERSION:-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-Standard_DS2_v2}"
DEPLOYMENT_NAME="${DEPLOYMENT_NAME:-blue}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
step() { echo -e "\n=== $* ==="; }

# --- Teardown: delete endpoint (stops billing) ------------------------------
if [[ "${TEARDOWN:-0}" == "1" ]]; then
    step "Tearing down endpoint '$ENDPOINT_NAME' (stops billing)"
    az ml online-endpoint delete --name "$ENDPOINT_NAME" \
        --resource-group "$RESOURCE_GROUP" --workspace-name "$WORKSPACE" --yes
    echo "Endpoint deleted. The registered model '$MODEL_NAME' is kept."
    exit 0
fi

# --- 0. (optional) rebuild bundle -------------------------------------------
if [[ "${REBUILD:-0}" == "1" ]]; then
    step "Rebuilding model bundle from MLflow"
    (cd "$PROJECT_ROOT" && uv run python deployment/export_bundle.py)
fi
[[ -f "$SCRIPT_DIR/model_bundle/routing.json" ]] || {
    echo "model_bundle/ not found. Run with REBUILD=1 first." >&2; exit 1; }

# --- 1. az defaults ----------------------------------------------------------
step "Setting az defaults: $RESOURCE_GROUP / $WORKSPACE"
az configure --defaults group="$RESOURCE_GROUP" workspace="$WORKSPACE"

# --- 2. register model -------------------------------------------------------
step "Registering model '$MODEL_NAME:$MODEL_VERSION'"
az ml model create --name "$MODEL_NAME" --version "$MODEL_VERSION" \
    --path "$SCRIPT_DIR/model_bundle" --type custom_model

# --- 3. create endpoint (idempotent) ----------------------------------------
step "Ensuring endpoint '$ENDPOINT_NAME' exists"
if ! az ml online-endpoint show --name "$ENDPOINT_NAME" >/dev/null 2>&1; then
    az ml online-endpoint create -f "$SCRIPT_DIR/endpoint.yml"
else
    echo "Endpoint already exists — skipping create."
fi

# --- 4. create/update deployment + route traffic ----------------------------
step "Creating deployment '$DEPLOYMENT_NAME' (instance: $INSTANCE_TYPE) — can take 10-20 min"
az ml online-deployment create -f "$SCRIPT_DIR/deployment.yml" \
    --set model="azureml:$MODEL_NAME:$MODEL_VERSION" instance_type="$INSTANCE_TYPE" \
    --all-traffic

# --- 5. smoke test -----------------------------------------------------------
if [[ "${SKIP_TEST:-0}" != "1" ]]; then
    step "Smoke test via example_input.json"
    az ml online-endpoint invoke --name "$ENDPOINT_NAME" \
        --request-file "$SCRIPT_DIR/example_input.json"
fi

URI="$(az ml online-endpoint show --name "$ENDPOINT_NAME" --query scoring_uri -o tsv)"
echo -e "\nDone. Scoring URI: $URI"
echo "Remember: the deployment bills continuously. Run with TEARDOWN=1 to stop."
