#!/usr/bin/env python3
"""Unit tests for device configuration validation."""

import unittest

from utils.config import validate_all_devices


class ConfigValidationTests(unittest.TestCase):
    def test_valid_config(self):
        valid_config = {
            "router1": {
                "ip": "192.168.1.1",
                "port": 22,
                "username": "admin",
                "auth": {"type": "password", "password": "secret123"},
            }
        }
        validate_all_devices(valid_config)

    def test_missing_ip(self):
        config = {
            "router1": {
                "port": 22,
                "username": "admin",
                "auth": {"type": "password", "password": "secret123"},
            }
        }
        with self.assertRaises(ValueError):
            validate_all_devices(config)

    def test_missing_auth(self):
        config = {"router1": {"ip": "192.168.1.1", "port": 22, "username": "admin"}}
        with self.assertRaises(ValueError):
            validate_all_devices(config)

    def test_invalid_auth_type(self):
        config = {
            "router1": {
                "ip": "192.168.1.1",
                "port": 22,
                "username": "admin",
                "auth": {"type": "invalid_type", "password": "secret123"},
            }
        }
        with self.assertRaises(ValueError):
            validate_all_devices(config)

    def test_ssh_key_missing_path(self):
        config = {
            "router1": {
                "ip": "192.168.1.1",
                "port": 22,
                "username": "admin",
                "auth": {"type": "ssh_key"},
            }
        }
        with self.assertRaises(ValueError):
            validate_all_devices(config)

    def test_invalid_port_type(self):
        config = {
            "router1": {
                "ip": "192.168.1.1",
                "port": "22",
                "username": "admin",
                "auth": {"type": "password", "password": "secret123"},
            }
        }
        with self.assertRaises(ValueError):
            validate_all_devices(config)

    def test_backward_compatibility_password_format(self):
        config = {
            "router1": {
                "ip": "192.168.1.1",
                "port": 22,
                "username": "admin",
                "password": "secret123",
            }
        }
        validate_all_devices(config)

    def test_multiple_devices_reports_all_invalid(self):
        config = {
            "router1": {
                "ip": "192.168.1.1",
                "port": 22,
                "username": "admin",
                "auth": {"type": "password", "password": "secret123"},
            },
            "router2": {
                "port": 22,
                "username": "admin",
                "auth": {"type": "ssh_key", "private_key_path": "/path/to/key"},
            },
            "router3": {
                "ip": "192.168.1.3",
                "port": "invalid",
                "username": "admin",
                "password": "secret",
            },
        }

        with self.assertRaises(ValueError) as ctx:
            validate_all_devices(config)

        error_str = str(ctx.exception)
        self.assertIn("router2", error_str)
        self.assertIn("router3", error_str)


if __name__ == "__main__":
    unittest.main()