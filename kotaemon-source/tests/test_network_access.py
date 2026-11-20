import base64
import json
import unittest

from network_access import (
    allowed_hosts,
    host_header_allowed,
    origin_allowed,
    validate_lan_ipv4,
    validate_public_origin,
    proxy_request_allowed,
    has_app_service_principal,
    app_service_request_allowed,
    request_within_boundary,
    WebSocketBoundaryMiddleware,
)


class NetworkAccessTests(unittest.TestCase):
    def test_lan_ip_must_be_private_ipv4(self):
        self.assertEqual(validate_lan_ipv4("192.168.50.10"), "192.168.50.10")
        for address in ("127.0.0.1", "169.254.1.2", "8.8.8.8", "::1", "invalid"):
            with self.subTest(address=address), self.assertRaises(ValueError):
                validate_lan_ipv4(address)

    def test_host_and_origin_must_match_exact_bound_addresses_and_port(self):
        hosts = allowed_hosts("192.168.50.10")
        self.assertTrue(host_header_allowed("192.168.50.10:7863", 7863, hosts))
        self.assertTrue(host_header_allowed("127.0.0.1:7863", 7863, hosts))
        self.assertFalse(host_header_allowed("192.168.50.11:7863", 7863, hosts))
        self.assertFalse(host_header_allowed("192.168.50.10:7864", 7863, hosts))
        self.assertFalse(host_header_allowed("192.168.50.10.evil:7863", 7863, hosts))
        self.assertTrue(origin_allowed("http://192.168.50.10:7863", 7863, hosts))
        self.assertFalse(origin_allowed("http://192.168.50.11:7863", 7863, hosts))
        self.assertFalse(origin_allowed("https://192.168.50.10:7863", 7863, hosts))

    def test_enterprise_proxy_is_explicit_https_and_loopback_only(self):
        origin = validate_public_origin("https://assistant.example.internal")
        self.assertTrue(proxy_request_allowed(
            "assistant.example.internal", origin, "127.0.0.1", origin
        ))
        self.assertFalse(proxy_request_allowed(
            "assistant.example.internal", origin, "10.1.2.3", origin
        ))
        self.assertFalse(proxy_request_allowed(
            "assistant.example.internal", "https://evil.example", "127.0.0.1", origin
        ))
        for value in ("http://assistant.example", "https://x.example/path", "https://user@x.example"):
            with self.assertRaises(ValueError):
                validate_public_origin(value)

    def test_app_service_requires_aad_principal_and_expected_origin(self):
        principal = base64.b64encode(json.dumps({
            "auth_typ": "aad",
            "claims": [{
                "typ": "http://schemas.microsoft.com/identity/claims/objectidentifier",
                "val": "user-object-id",
            }],
        }).encode()).decode()
        self.assertTrue(has_app_service_principal(principal))
        self.assertFalse(has_app_service_principal("not-base64"))
        self.assertFalse(has_app_service_principal("\u00e9"))
        self.assertFalse(has_app_service_principal(base64.b64encode(
            b'{"auth_typ":"aad","claims":[]}'
        ).decode()))
        origin = "https://rfq.example.com"
        self.assertTrue(app_service_request_allowed(
            "rfq.example.com", origin, origin, authenticated=True
        ))
        self.assertFalse(app_service_request_allowed(
            "rfq.example.com", "https://evil.example", origin, authenticated=True
        ))
        self.assertFalse(app_service_request_allowed(
            "rfq.example.com", origin, origin, authenticated=False
        ))
        self.assertFalse(request_within_boundary(
            host="127.0.0.1:8000",
            origin="http://127.0.0.1:8000",
            client_ip="203.0.113.10",
            port=8000,
            hosts=allowed_hosts(),
            public_origin=origin,
            app_service_mode=True,
        ))
        self.assertTrue(request_within_boundary(
            host="rfq.example.com",
            origin=origin,
            client_ip="10.0.0.2",
            port=8000,
            hosts=allowed_hosts(),
            public_origin=origin,
            app_service_mode=True,
            principal_header=principal,
        ))

    def test_company_ip_only_mode_keeps_host_and_origin_checks(self):
        options = {
            "client_ip": "10.0.0.2",
            "port": 8000,
            "hosts": allowed_hosts(),
            "public_origin": "https://rfq.example.com",
            "app_service_mode": True,
            "app_service_ip_restricted": True,
        }
        self.assertTrue(request_within_boundary(
            host="rfq.example.com", origin="https://rfq.example.com", **options
        ))
        self.assertFalse(request_within_boundary(
            host="rfq.example.com", origin="https://attacker.example", **options
        ))
        self.assertFalse(request_within_boundary(
            host="attacker.example", origin="https://rfq.example.com", **options
        ))
        self.assertFalse(request_within_boundary(
            host="rfq.example.com", origin="https://rfq.example.com",
            **{**options, "app_service_ip_restricted": False}
        ))


class WebSocketBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_websocket_checks_the_same_boundary_as_http(self):
        reached = []
        messages = []

        async def app(scope, receive, send):
            reached.append(scope)

        async def send(message):
            messages.append(message)

        middleware = WebSocketBoundaryMiddleware(
            app, port=8000, hosts=allowed_hosts(),
            public_origin="https://rfq.example.com", app_service_mode=True,
            app_service_ip_restricted=True,
        )
        scope = {
            "type": "websocket",
            "client": ("10.0.0.2", 1234),
            "headers": [
                (b"host", b"rfq.example.com"),
                (b"origin", b"https://rfq.example.com"),
            ],
        }
        await middleware(scope, None, send)
        self.assertEqual(len(reached), 1)
        self.assertEqual(messages, [])

        scope["headers"][1] = (b"origin", b"https://attacker.example")
        await middleware(scope, None, send)
        self.assertEqual(len(reached), 1)
        self.assertEqual(messages, [{"type": "websocket.close", "code": 1008}])

        scope["headers"] = [(b"host", b"rfq.example.com"), (b"host", b"attacker.example")]
        await middleware(scope, None, send)
        self.assertEqual(len(reached), 1)
        self.assertEqual(messages[-1], {"type": "websocket.close", "code": 1008})


if __name__ == "__main__":
    unittest.main()
