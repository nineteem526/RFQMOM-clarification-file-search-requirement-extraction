#!/usr/bin/env bash
set -euo pipefail

: "${AZURE_SUBSCRIPTION_ID:?Set the Azure subscription ID secret.}"
: "${AZURE_RESOURCE_GROUP:?Set the Web App resource group variable.}"
: "${AZURE_WEBAPP_NAME:?Set the Web App name variable.}"
: "${AZURE_ACR_NAME:?Set the ACR name variable.}"
: "${AZURE_ACR_RESOURCE_GROUP:?Set the ACR resource group variable.}"
: "${GITHUB_SHA:?}"
: "${GITHUB_RUN_ID:?}"
: "${GITHUB_RUN_ATTEMPT:?}"
: "${RUNNER_TEMP:?}"

if [ "${GITHUB_REF:-}" != "refs/heads/main" ]; then
    echo "Deployment is permitted only from main." >&2
    exit 1
fi
if [[ ! "$GITHUB_SHA" =~ ^[0-9a-f]{40}$ ]] \
    || [[ ! "$GITHUB_RUN_ID" =~ ^[0-9]+$ ]] \
    || [[ ! "$GITHUB_RUN_ATTEMPT" =~ ^[0-9]+$ ]]; then
    echo "Invalid GitHub commit or run identity." >&2
    exit 1
fi
if [[ ! "$AZURE_SUBSCRIPTION_ID" =~ ^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$ ]] \
    || [[ ! "$AZURE_RESOURCE_GROUP" =~ ^[[:alnum:]_.()-]+$ ]] \
    || [[ ! "$AZURE_ACR_RESOURCE_GROUP" =~ ^[[:alnum:]_.()-]+$ ]] \
    || [[ ! "$AZURE_WEBAPP_NAME" =~ ^[[:alnum:]-]+$ ]] \
    || [[ ! "$AZURE_ACR_NAME" =~ ^[[:alnum:]]+$ ]]; then
    echo "Invalid Azure subscription or resource name." >&2
    exit 1
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
checks="$(mktemp -d "$RUNNER_TEMP/rfq-azure-check.XXXXXX")"
cleanup() {
    status=$?
    if ! rm -rf -- "$checks"; then
        echo "Failed to clean up temporary deployment metadata." >&2
        status=1
    fi
    exit "$status"
}
trap cleanup EXIT
az webapp show --resource-group "$AZURE_RESOURCE_GROUP" --name "$AZURE_WEBAPP_NAME" \
    --query '{httpsOnly:httpsOnly}' --output json > "$checks/site.json"
az webapp config show --resource-group "$AZURE_RESOURCE_GROUP" --name "$AZURE_WEBAPP_NAME" \
    --query '{acrUseManagedIdentityCreds:acrUseManagedIdentityCreds}' --output json > "$checks/container.json"
site_config="https://management.azure.com/subscriptions/$AZURE_SUBSCRIPTION_ID/resourceGroups/$AZURE_RESOURCE_GROUP/providers/Microsoft.Web/sites/$AZURE_WEBAPP_NAME/config"
auth_url="$site_config/authsettingsV2?api-version=2024-11-01"
az rest --method get --url "$auth_url" --output json > "$checks/auth.json"
python3 -B "$root/environment-setup/check_appservice_configuration.py" \
    --site "$checks/site.json" --container "$checks/container.json" \
    --auth "$checks/auth.json"

registry="$(az acr show --name "$AZURE_ACR_NAME" --resource-group "$AZURE_ACR_RESOURCE_GROUP" \
    --query loginServer --output tsv)"
if [[ ! "$registry" =~ ^[A-Za-z0-9][A-Za-z0-9.-]*$ ]]; then
    echo "ACR returned an invalid login server." >&2
    exit 1
fi
tag="$GITHUB_SHA-$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT"
image="$registry/rfq-kotaemon:$tag"
az acr login --name "$AZURE_ACR_NAME" --output none
docker tag "rfq-kotaemon-ci:$tag" "$image"
docker push "$image"
body="$(printf '{"properties":{"linuxFxVersion":"DOCKER|%s"}}' "$image")"
az rest --method patch --url "$site_config/web?api-version=2024-11-01" \
    --body "$body" --output none

echo "App Service image configuration updated. Container startup and live access still need verification."
