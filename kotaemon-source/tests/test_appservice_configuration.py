from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "environment-setup"))
from check_appservice_configuration import configuration_errors


def valid_configuration():
    return (
        {"properties": {
            "platform": {"enabled": True},
            "globalValidation": {
                "requireAuthentication": True,
                "unauthenticatedClientAction": "RedirectToLoginPage",
                "excludedPaths": [],
            },
            "identityProviders": {"azureActiveDirectory": {
                "enabled": True,
                "registration": {
                    "clientId": "fixture",
                    "openIdIssuer": "https://login.example.invalid/fixture/v2.0",
                },
            }},
        }},
        {"httpsOnly": True},
        {"acrUseManagedIdentityCreds": True},
    )


class AppServiceConfigurationTests(unittest.TestCase):
    def test_prepared_configuration_is_accepted(self):
        self.assertEqual(configuration_errors(*valid_configuration()), [])

    def test_unprotected_or_file_based_authentication_is_blocked(self):
        for mutate in (
            lambda p: p["platform"].update(enabled=False),
            lambda p: p["platform"].update(configFilePath="/external.json"),
            lambda p: p["globalValidation"].update(requireAuthentication=False),
            lambda p: p["globalValidation"].update(unauthenticatedClientAction="AllowAnonymous"),
            lambda p: p["globalValidation"].update(excludedPaths=["/"]),
            lambda p: p["identityProviders"]["azureActiveDirectory"].update(enabled=False),
            lambda p: p["identityProviders"]["azureActiveDirectory"].pop("enabled"),
            lambda p: p["identityProviders"]["azureActiveDirectory"]["registration"].pop("openIdIssuer"),
        ):
            auth, site, container = valid_configuration()
            mutate(auth["properties"])
            with self.subTest(auth=auth):
                self.assertTrue(configuration_errors(auth, site, container))

    def test_https_and_managed_identity_are_required(self):
        auth, _, _ = valid_configuration()
        errors = configuration_errors(
            auth, {"httpsOnly": False}, {"acrUseManagedIdentityCreds": False},
        )
        self.assertEqual(len(errors), 2)


if __name__ == "__main__":
    unittest.main()
