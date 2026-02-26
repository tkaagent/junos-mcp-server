#!/usr/bin/env python3
"""Unit tests for handle_get_router_list() and data redaction."""

import json
import unittest
from unittest.mock import MagicMock

import jmcp


class GetRouterListTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._original_devices = jmcp.devices.copy()
        self.context = MagicMock()

    def tearDown(self):
        jmcp.devices = self._original_devices

    async def test_empty_devices(self):
        jmcp.devices = {}
        result = await jmcp.handle_get_router_list({}, self.context)
        self.assertEqual(len(result), 1)
        self.assertEqual(json.loads(result[0].text), {})

    async def test_password_and_ssh_key_redaction(self):
        jmcp.devices = {
            "router-1": {
                "ip": "192.168.1.1",
                "port": 22,
                "username": "admin",
                "auth": {"type": "password", "password": "secret123"},
            },
            "router-2": {
                "ip": "192.168.1.2",
                "port": 22,
                "username": "admin",
                "auth": {"type": "ssh_key", "private_key_path": "/path/to/key.pem"},
            },
        }

        result = await jmcp.handle_get_router_list({}, self.context)
        output = json.loads(result[0].text)

        self.assertEqual(output["router-1"]["auth"]["type"], "password")
        self.assertNotIn("password", output["router-1"]["auth"])

        self.assertEqual(output["router-2"]["auth"]["type"], "ssh_key")
        self.assertNotIn("private_key_path", output["router-2"]["auth"])

    async def test_ssh_config_removed_and_custom_fields_kept(self):
        jmcp.devices = {
            "router-3": {
                "ip": "192.168.1.3",
                "port": 22,
                "username": "admin",
                "ssh_config": "/home/user/.ssh/config_jumphost",
                "role": "pe",
                "group": "ISP",
                "auth": {"type": "password", "password": "secret123"},
            }
        }

        result = await jmcp.handle_get_router_list({}, self.context)
        output = json.loads(result[0].text)
        router = output["router-3"]

        self.assertNotIn("ssh_config", router)
        self.assertEqual(router["role"], "pe")
        self.assertEqual(router["group"], "ISP")
        self.assertNotIn("password", router["auth"])

    async def test_json_pretty_and_source_immutability(self):
        jmcp.devices = {
            "test-router": {
                "ip": "10.0.0.1",
                "port": 22,
                "username": "test",
                "auth": {"type": "password", "password": "test123"},
            }
        }
        before = json.dumps(jmcp.devices, sort_keys=True)

        result = await jmcp.handle_get_router_list({}, self.context)
        output_text = result[0].text

        json.loads(output_text)
        self.assertIn("\n", output_text)
        self.assertIn("  ", output_text)

        after = json.dumps(jmcp.devices, sort_keys=True)
        self.assertEqual(before, after)
        self.assertIn("password", jmcp.devices["test-router"]["auth"])


if __name__ == "__main__":
    unittest.main()
