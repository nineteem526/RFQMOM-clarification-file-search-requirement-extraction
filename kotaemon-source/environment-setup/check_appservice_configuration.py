"""Fail closed when required App Service deployment settings are absent."""

import argparse
import json
from pathlib import Path
import sys


def configuration_errors(auth, site, container):
    errors = []
    if not isinstance(auth, dict):
        return ["Authentication configuration is not an object."]
    properties = auth.get("properties")
    if not isinstance(properties, dict):
        return ["Authentication configuration has no properties."]
    platform = properties.get("platform") or {}
    validation = properties.get("globalValidation") or {}
    providers = properties.get("identityProviders") or {}
    if not all(isinstance(item, dict) for item in (platform, validation, providers)):
        return ["Authentication configuration has an invalid shape."]
    aad = providers.get("azureActiveDirectory") or {}
    if not isinstance(aad, dict):
        return ["Microsoft Entra ID configuration has an invalid shape."]
    registration = aad.get("registration") or {}
    if not isinstance(registration, dict):
        return ["Microsoft Entra ID registration has an invalid shape."]
    if platform.get("enabled") is not True:
        errors.append("Enable App Service Authentication.")
    if platform.get("configFilePath"):
        errors.append("File-based authentication configuration must be reviewed separately.")
    if validation.get("requireAuthentication") is not True:
        errors.append("Require authentication for all requests.")
    if validation.get("unauthenticatedClientAction") not in {
        "RedirectToLoginPage", "Return401", "Return403",
    }:
        errors.append("Reject or redirect unauthenticated requests.")
    if validation.get("excludedPaths"):
        errors.append("Remove unauthenticated excluded paths.")
    if (aad.get("enabled") is not True
            or not isinstance(registration.get("clientId"), str)
            or not registration["clientId"].strip()
            or not isinstance(registration.get("openIdIssuer"), str)
            or not registration["openIdIssuer"].strip()):
        errors.append("Configure the Microsoft Entra ID authentication provider.")
    if not isinstance(site, dict) or site.get("httpsOnly") is not True:
        errors.append("Enable HTTPS only on the Web App.")
    if (not isinstance(container, dict)
            or container.get("acrUseManagedIdentityCreds") is not True):
        errors.append("Configure the Web App to pull from ACR with a managed identity.")
    return errors


def main():
    parser = argparse.ArgumentParser()
    for name in ("auth", "site", "container"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    try:
        data = [
            json.loads(getattr(args, name).read_text(encoding="utf-8"))
            for name in ("auth", "site", "container")
        ]
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        print(f"Cannot read App Service configuration: {error}", file=sys.stderr)
        return 1
    errors = configuration_errors(*data)
    if errors:
        for error in errors:
            print(f"Deployment blocked: {error}", file=sys.stderr)
        return 1
    print("Required App Service authentication, HTTPS and registry identity settings are present.")
    print("App settings, group assignment, source mount and single-instance settings require separate verification.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
