#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf -- "$work"' EXIT
mkdir -p "$work/bin"
export RFQ_TEST_EVENTS="$work/events"
export RUNNER_TEMP="$work"
export AZURE_SUBSCRIPTION_ID="00000000-0000-0000-0000-000000000000"
export AZURE_RESOURCE_GROUP="rfq-test"
export AZURE_WEBAPP_NAME="rfq-web"
export AZURE_ACR_NAME="rfqregistry"
export AZURE_ACR_RESOURCE_GROUP="rfq-registry"
export GITHUB_REF="refs/heads/main"
export GITHUB_SHA="1111111111111111111111111111111111111111"
export GITHUB_RUN_ID="12345"
export GITHUB_RUN_ATTEMPT="1"

cat > "$work/bin/az" <<'MOCK'
#!/usr/bin/env bash
set -euo pipefail
printf 'az' >> "$RFQ_TEST_EVENTS"
printf ' <%s>' "$@" >> "$RFQ_TEST_EVENTS"
printf '\n' >> "$RFQ_TEST_EVENTS"
case "$1 $2 ${3:-}" in
    "webapp show --resource-group")
        printf '%s\n' '{"httpsOnly":true}'
        ;;
    "webapp config show")
        printf '%s\n' '{"acrUseManagedIdentityCreds":true}'
        ;;
    "rest --method get")
        enabled=true
        if [ "${RFQ_TEST_UNPROTECTED:-false}" = true ]; then enabled=false; fi
        printf '{"properties":{"platform":{"enabled":%s},"globalValidation":{"requireAuthentication":true,"unauthenticatedClientAction":"RedirectToLoginPage"},"identityProviders":{"azureActiveDirectory":{"enabled":true,"registration":{"clientId":"fixture","openIdIssuer":"https://login.example.invalid"}}}}}\n' "$enabled"
        ;;
    "acr show --name")
        printf '%s\n' 'rfqregistry.azurecr.io'
        ;;
    "acr login --name"|"rest --method patch")
        ;;
    *)
        echo "Unexpected Azure command." >&2
        exit 1
        ;;
esac
MOCK

cat > "$work/bin/docker" <<'MOCK'
#!/usr/bin/env bash
set -euo pipefail
printf 'docker' >> "$RFQ_TEST_EVENTS"
printf ' <%s>' "$@" >> "$RFQ_TEST_EVENTS"
printf '\n' >> "$RFQ_TEST_EVENTS"
case "$1" in tag|push) ;; *) exit 1 ;; esac
MOCK
chmod +x "$work/bin/az" "$work/bin/docker"
export PATH="$work/bin:$PATH"

script="$root/environment-setup/deploy_appservice.sh"
export RFQ_TEST_UNPROTECTED=true
if bash "$script" > "$work/unprotected.log" 2>&1; then
    echo "Deployment accepted disabled authentication." >&2
    exit 1
fi
if grep -q '^docker' "$RFQ_TEST_EVENTS"; then
    echo "Image was pushed before authentication checks passed." >&2
    exit 1
fi
if compgen -G "$work/rfq-azure-check.*" >/dev/null; then
    echo "Temporary metadata was not removed after a rejected deployment." >&2
    exit 1
fi
unset RFQ_TEST_UNPROTECTED

: > "$RFQ_TEST_EVENTS"
if GITHUB_REF="refs/heads/feature" bash "$script" > "$work/branch.log" 2>&1; then
    echo "Deployment accepted a non-main branch." >&2
    exit 1
fi
if [ -s "$RFQ_TEST_EVENTS" ]; then
    echo "Deployment contacted Azure for a rejected branch." >&2
    exit 1
fi

bash "$script"
if ! grep -Fq "\"linuxFxVersion\":\"DOCKER|rfqregistry.azurecr.io/rfq-kotaemon:$GITHUB_SHA-$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT\"" "$RFQ_TEST_EVENTS"; then
    echo "Deployment did not set the expected image." >&2
    exit 1
fi
if grep -Eq 'appsettings|config> <container|registry-password|registry-user' "$RFQ_TEST_EVENTS"; then
    echo "Deployment unexpectedly accessed app settings or registry credentials." >&2
    exit 1
fi
if compgen -G "$work/rfq-azure-check.*" >/dev/null; then
    echo "Temporary metadata was not removed after deployment." >&2
    exit 1
fi
echo "Deployment command checks passed."
