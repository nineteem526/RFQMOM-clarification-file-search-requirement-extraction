param(
    [Parameter(Mandatory = $true)]
    [string]$ResourceGroup,

    [Parameter(Mandatory = $true)]
    [string]$WebAppName,

    [Parameter(Mandatory = $true)]
    [string]$PackagePath,

    [Parameter(Mandatory = $true)]
    [string]$SubscriptionId,

    [string]$AzureCli = "az.cmd",

    [switch]$CompanyIpOnly,

    [string]$CompanyEgressCidr,

    [string]$RuntimeRoot = "/home/rfq"
)

$ErrorActionPreference = "Stop"

if ($AzureCli -eq "az.cmd") {
    $command = Get-Command az.cmd -ErrorAction SilentlyContinue
    if (-not $command) {
        throw "Azure CLI was not found. Set -AzureCli to the full path of az.cmd."
    }
    $AzureCli = $command.Source
} elseif (-not (Test-Path -LiteralPath $AzureCli -PathType Leaf)) {
    throw "Azure CLI was not found at the supplied -AzureCli path."
}

$package = [System.IO.Path]::GetFullPath($PackagePath)
if (-not (Test-Path -LiteralPath $package -PathType Leaf)) {
    throw "Deployment package was not found: $package"
}
if ([System.IO.Path]::GetExtension($package) -ne ".zip") {
    throw "Deployment package must be a ZIP file."
}
if ($RuntimeRoot -notmatch '^/home/[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)*$') {
    throw "RuntimeRoot must be a safe absolute directory under /home."
}

$selectedSubscription = & $AzureCli account show `
    --query id --output tsv --only-show-errors
if ($LASTEXITCODE -ne 0) {
    throw "Azure CLI has no active signed-in account."
}
if ($selectedSubscription.Trim() -ne $SubscriptionId) {
    throw "Azure CLI is using a different subscription; select the approved test subscription first."
}

$site = & $AzureCli webapp show `
    --resource-group $ResourceGroup --name $WebAppName `
    --subscription $SubscriptionId `
    --query "{id:id,kind:kind,httpsOnly:httpsOnly,serverFarmId:appServicePlanId,host:defaultHostName,publicNetworkAccess:publicNetworkAccess}" `
    --output json --only-show-errors | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) {
    throw "Could not read the target Web App."
}
if ($site.kind -notmatch "linux" -or $site.httpsOnly -ne $true) {
    throw "Target must be a Linux Web App with HTTPS Only enabled."
}

$plan = & $AzureCli appservice plan show `
    --ids $site.serverFarmId --subscription $SubscriptionId `
    --query "{linux:properties.reserved,workers:properties.numberOfWorkers,skuWorkers:sku.capacity,sku:sku.name}" `
    --output json --only-show-errors | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) {
    throw "Could not read the target App Service plan."
}
if ($plan.linux -ne $true -or $plan.workers -ne 1 -or $plan.skuWorkers -ne 1) {
    throw "Target plan must be Linux with exactly one worker."
}

$config = & $AzureCli webapp config show `
    --resource-group $ResourceGroup --name $WebAppName `
    --subscription $SubscriptionId `
    --query "{runtime:linuxFxVersion,mainDefault:ipSecurityRestrictionsDefaultAction,scmDefault:scmIpSecurityRestrictionsDefaultAction,scmUsesMain:scmIpSecurityRestrictionsUseMain,main:ipSecurityRestrictions}" `
    --output json --only-show-errors | ConvertFrom-Json
if ($LASTEXITCODE -ne 0 -or $config.runtime -ne "PYTHON|3.11") {
    throw "Target Web App must use the built-in Linux Python 3.11 runtime."
}

$authUrl = "https://management.azure.com$($site.id)/config/authsettingsV2?api-version=2022-03-01"
$auth = & $AzureCli rest --method get --url $authUrl `
    --query "{enabled:properties.platform.enabled,configFilePath:properties.platform.configFilePath,requireAuthentication:properties.globalValidation.requireAuthentication,action:properties.globalValidation.unauthenticatedClientAction,excludedPaths:properties.globalValidation.excludedPaths,entraEnabled:properties.identityProviders.azureActiveDirectory.enabled,clientId:properties.identityProviders.azureActiveDirectory.registration.clientId,issuer:properties.identityProviders.azureActiveDirectory.registration.openIdIssuer}" `
    --output json --only-show-errors | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) {
    throw "Could not verify the Web App's Entra authentication configuration."
}
$accessMode = "entra"
if ($CompanyIpOnly) {
    $address = $null
    if (-not $CompanyEgressCidr -or $CompanyEgressCidr -notmatch '^([^/]+)/32$' -or
        -not [System.Net.IPAddress]::TryParse($Matches[1], [ref]$address) -or
        $address.AddressFamily -ne [System.Net.Sockets.AddressFamily]::InterNetwork -or
        $CompanyEgressCidr -ne "$($address.ToString())/32") {
        throw "Company IP-only mode requires one canonical IPv4 /32 address."
    }
    if ($site.publicNetworkAccess -ne "Enabled" -or $config.mainDefault -ne "Deny" -or
        $config.scmDefault -ne "Deny" -or $config.scmUsesMain -ne $true) {
        throw "Company IP-only mode requires deny-by-default web and shared SCM restrictions."
    }
    $allowed = @($config.main | Where-Object { $_.action -eq "Allow" })
    if ($allowed.Count -eq 0 -or @($allowed | Where-Object {
        $_.ipAddress -ne $CompanyEgressCidr -or $_.vnetSubnetResourceId -or
        ($_.tag -and $_.tag -ne "Default")
    }).Count -gt 0) {
        throw "Web and SCM must allow only the confirmed company egress IP."
    }
    if ($auth.enabled -ne $false -or $auth.configFilePath) {
        throw "Existing App Service authentication must be reviewed; it will not be disabled automatically."
    }
    $accessMode = "company-ip"
    Write-Warning "No user authentication: everyone using the allowed egress can access and modify the project."
} else {
    if ($auth.enabled -ne $true -or $auth.configFilePath -or
        $auth.requireAuthentication -ne $true -or
        $auth.action -notin @("RedirectToLoginPage", "Return401", "Return403") -or
        @($auth.excludedPaths).Count -gt 0 -or $auth.entraEnabled -ne $true -or
        -not $auth.clientId -or -not $auth.issuer) {
        throw "Configure and verify required Entra authentication before deploying the app."
    }
}

$settings = @(
    "SCM_DO_BUILD_DURING_DEPLOYMENT=true",
    "PORT=8000",
    "RFQ_APP_SERVICE=true",
    "RFQ_APP_SERVICE_ACCESS_MODE=$accessMode",
    "RFQ_SINGLE_PROJECT_MODE=true",
    "RFQ_RUNTIME_HOME=$RuntimeRoot/data",
    "RFQ_AUTHORIZED_SOURCE=$RuntimeRoot/source",
    "RFQ_PROJECT_ID=rfq-poc",
    "RFQ_PUBLIC_ORIGIN=https://$($site.host)"
)
& $AzureCli webapp config appsettings set `
    --resource-group $ResourceGroup --name $WebAppName `
    --subscription $SubscriptionId --settings $settings `
    --output none --only-show-errors
if ($LASTEXITCODE -ne 0) {
    throw "Could not configure App Service runtime settings."
}

& $AzureCli webapp config set `
    --resource-group $ResourceGroup --name $WebAppName `
    --subscription $SubscriptionId `
    --startup-file "bash start-appservice.sh" `
    --output none --only-show-errors
if ($LASTEXITCODE -ne 0) {
    throw "Could not configure the Python runtime and startup command."
}

& $AzureCli webapp deploy `
    --resource-group $ResourceGroup --name $WebAppName `
    --subscription $SubscriptionId --src-path $package `
    --type zip --clean false --async true --track-status false --restart true `
    --only-show-errors
if ($LASTEXITCODE -ne 0) {
    throw "ZIP deployment request failed or timed out. Check the latest deployment status before retrying."
}

Write-Output "Source package upload accepted. The Oryx build may still be running; verify its status and app startup before using the site."
