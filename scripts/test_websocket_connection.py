#!/usr/bin/env python3
"""
AutoVoice WebSocket Connection Test Script
Tests Socket.IO WebSocket client functionality as described in Phase 3 plan.
"""

import json
import sys
import time
import threading
from typing import Dict, Any

try:
    import socketio
    import websockets
except ImportError as e:
    print(f"ERROR: Required packages not available: {e}")
    print("Install with: pip install python-socketio websockets")
    sys.exit(1)

class WebSocketTestClient:
    """Socket.IO test client for AutoVoice WebSocket interface."""

    def __init__(self, url: str = "http://localhost:5000"):
        self.url = url
        self.sio = socketio.Client()
        self.connection_success = False
        self.connected_received = False
        self.status_received = False
        self.status_data = None
        self.test_results = {
            "connection": False,
            "handshake": False,
            "status_request": False,
            "response_parsing": False,
            "cleanup": False,
            "errors": []
        }
        self.event_received = threading.Event()

        # Register event handlers
        self._register_handlers()

    def _register_handlers(self):
        """Register Socket.IO event handlers"""

        @self.sio.on('connected')
        def on_connected(data):
            print(f"Received 'connected' event: {data}")
            self.connected_received = True
            self.test_results["handshake"] = True
            self.event_received.set()

        @self.sio.on('status')
        def on_status(data):
            print(f"Received 'status' event: {data}")
            self.status_received = True
            self.status_data = data
            self.test_results["status_request"] = True
            self.test_results["response_parsing"] = True
            self.event_received.set()

        @self.sio.on('connect')
        def on_connect():
            print("Socket.IO connection established")
            self.connection_success = True
            self.test_results["connection"] = True

        @self.sio.on('disconnect')
        def on_disconnect():
            print("Socket.IO disconnected")

        @self.sio.on('error')
        def on_error(data):
            self.log_error(f"Socket.IO error: {data}")

    def log_error(self, message: str):
        """Log an error and mark test as failed."""
        self.test_results["errors"].append(message)
        print(f"ERROR: {message}")

    def log_success(self, message: str):
        """Log a success message."""
        print(f"SUCCESS: {message}")

    def connect_and_test(self, timeout: int = 30) -> bool:
        """Execute complete Socket.IO test sequence."""
        print(f"Testing Socket.IO connection to AutoVoice server at {self.url}...")

        try:
            # Test 1: Connect to Socket.IO server
            print("Connecting to Socket.IO server...")
            self.sio.connect(self.url)

            if not self.connection_success:
                self.log_error("Failed to establish Socket.IO connection")
                return False

            self.log_success("Socket.IO connection established")

            # Test 2: Wait for 'connected' event
            print("Waiting for 'connected' event...")
            self.event_received.clear()
            if not self.event_received.wait(timeout=10.0):
                self.log_error("Timeout waiting for 'connected' event")
                return False

            if not self.connected_received:
                self.log_error("Did not receive 'connected' event")
                return False

            self.log_success("Received 'connected' event")

            # Test 3: Send get_status request
            print("Sending 'get_status' event...")
            self.event_received.clear()
            self.sio.emit('get_status')

            # Test 4: Wait for status response
            print("Waiting for 'status' event...")
            if not self.event_received.wait(timeout=10.0):
                self.log_error("Timeout waiting for 'status' event")
                return False

            if not self.status_received:
                self.log_error("Did not receive 'status' event")
                return False

            self.log_success("Received 'status' event")

            # Validate status response structure
            if self.status_data:
                print(f"Status data: {self.status_data}")
                # Check for expected fields
                if 'timestamp' in self.status_data:
                    self.log_success("Status response contains timestamp")
                if 'capabilities' in self.status_data or 'metrics' in self.status_data:
                    self.log_success("Status response contains server info")

            # Test 5: Clean disconnect
            print("Initiating clean disconnect...")
            self.sio.disconnect()
            self.test_results["cleanup"] = True
            self.log_success("Clean disconnect completed")

            return True

        except socketio.exceptions.ConnectionError as e:
            self.log_error(f"Socket.IO connection failed: {str(e)}")
            return False
        except Exception as e:
            self.log_error(f"Unexpected error during Socket.IO test: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            # Ensure disconnection
            if self.sio.connected:
                try:
                    self.sio.disconnect()
                except:
                    pass

    def get_summary(self) -> Dict[str, Any]:
        """Get test summary."""
        all_tests_passed = all([
            self.test_results["connection"],
            self.test_results["handshake"],
            self.test_results["status_request"],
            self.test_results["response_parsing"],
            self.test_results["cleanup"]
        ])

        return {
            "success": all_tests_passed,
            "timestamp": time.time(),
            "results": self.test_results,
            "summary": {
                "total_tests": 5,
                "passed": sum(v for k, v in self.test_results.items() if isinstance(v, bool) and k != "errors"),
                "failed": len(self.test_results.get("errors", []))
            }
        }


class RawWebSocketTestClient:
    """Raw WebSocket test client for AutoVoice WebSocket interface."""

    def __init__(self, url: str = "ws://localhost:5000"):
        self.url = url.replace("http://", "ws://").replace("https://", "wss://")
        self.connection_success = False
        self.status_received = False
        self.status_data = None
        self.test_results = {
            "connection": False,
            "handshake": False,
            "status_request": False,
            "response_parsing": False,
            "cleanup": False,
            "errors": []
        }

    def log_error(self, message: str):
        """Log an error and mark test as failed."""
        self.test_results["errors"].append(message)
        print(f"ERROR: {message}")

    def log_success(self, message: str):
        """Log a success message."""
        print(f"SUCCESS: {message}")

    async def connect_and_test_async(self, timeout: int = 30) -> bool:
        """Execute complete raw WebSocket test sequence."""
        print(f"Testing raw WebSocket connection to AutoVoice server at {self.url}...")

        try:
            # Test 1: Connect to WebSocket server
            print("Connecting to WebSocket server...")
            async with websockets.connect(self.url) as websocket:
                self.connection_success = True
                self.test_results["connection"] = True
                self.log_success("WebSocket connection established")

                # Test 2: Send handshake message
                handshake_msg = json.dumps({"type": "handshake", "client": "test_client"})
                await websocket.send(handshake_msg)
                print("Sent handshake message")

                # Test 3: Wait for handshake response
                response = await websocket.recv()
                response_data = json.loads(response)
                print(f"Received handshake response: {response_data}")

                if response_data.get("type") == "handshake_ack":
                    self.test_results["handshake"] = True
                    self.log_success("Handshake completed")
                else:
                    self.log_error("Invalid handshake response")
                    return False

                # Test 4: Send get_status request
                status_msg = json.dumps({"type": "get_status"})
                await websocket.send(status_msg)
                print("Sent get_status request")

                # Test 5: Wait for status response
                response = await websocket.recv()
                response_data = json.loads(response)
                print(f"Received status response: {response_data}")

                if response_data.get("type") == "status":
                    self.status_received = True
                    self.status_data = response_data.get("data", response_data)
                    self.test_results["status_request"] = True
                    self.test_results["response_parsing"] = True
                    self.log_success("Received status response")

                    # Validate status response structure
                    if self.status_data:
                        if 'timestamp' in self.status_data:
                            self.log_success("Status response contains timestamp")
                        if 'capabilities' in self.status_data or 'metrics' in self.status_data:
                            self.log_success("Status response contains server info")
                else:
                    self.log_error("Invalid status response")
                    return False

                # Test 6: Clean disconnect
                await websocket.close()
                self.test_results["cleanup"] = True
                self.log_success("Clean disconnect completed")

                return True

        except websockets.exceptions.ConnectionClosedError as e:
            self.log_error(f"WebSocket connection closed: {str(e)}")
            return False
        except Exception as e:
            self.log_error(f"Unexpected error during WebSocket test: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def connect_and_test(self, timeout: int = 30) -> bool:
        """Synchronous wrapper for async test."""
        import asyncio
        try:
            return asyncio.run(self.connect_and_test_async(timeout))
        except Exception as e:
            self.log_error(f"Failed to run async WebSocket test: {str(e)}")
            return False

    def get_summary(self) -> Dict[str, Any]:
        """Get test summary."""
        all_tests_passed = all([
            self.test_results["connection"],
            self.test_results["handshake"],
            self.test_results["status_request"],
            self.test_results["response_parsing"],
            self.test_results["cleanup"]
        ])

        return {
            "success": all_tests_passed,
            "timestamp": time.time(),
            "results": self.test_results,
            "summary": {
                "total_tests": 5,
                "passed": sum(v for k, v in self.test_results.items() if isinstance(v, bool) and k != "errors"),
                "failed": len(self.test_results.get("errors", []))
            }
        }

def main():
    """Main test execution."""
    import argparse

    parser = argparse.ArgumentParser(description="AutoVoice WebSocket Connection Test")
    parser.add_argument("--url", default="http://localhost:5000",
                       help="WebSocket server URL (default: http://localhost:5000)")
    parser.add_argument("--timeout", type=int, default=30,
                       help="Test timeout in seconds")
    parser.add_argument("--json-output", help="Output results as JSON to file")
    parser.add_argument("--transport", choices=["socketio", "websocket"], default="socketio",
                       help="WebSocket transport to use (default: socketio)")

    args = parser.parse_args()

    # Choose client based on transport
    if args.transport == "socketio":
        client = WebSocketTestClient(args.url)
        transport_name = "Socket.IO"
    else:  # websocket
        client = RawWebSocketTestClient(args.url)
        transport_name = "Raw WebSocket"

    start_time = time.time()

    print(f"\n🧪 AutoVoice {transport_name} Connection Test")
    print(f"URL: {args.url}")
    print(f"Transport: {args.transport}")
    print(f"Timeout: {args.timeout}s")
    print("-" * 50)

    success = client.connect_and_test(timeout=args.timeout)

    elapsed = time.time() - start_time
    print("-" * 50)
    print(f"Elapsed time: {elapsed:.2f}s")

    summary = client.get_summary()

    # Always write JSON output if requested BEFORE exiting
    if args.json_output:
        with open(args.json_output, 'w') as f:
            json.dump(summary, f, indent=2)

    if success:
        print("✅ WebSocket tests completed successfully")
        print(f"   Tests passed: {summary['summary']['passed']}/{summary['summary']['total_tests']}")

        # Print detailed results
        for test_name, passed in summary["results"].items():
            if test_name != "errors" and isinstance(passed, bool):
                status = "✅" if passed else "❌"
                print(f"   {status} {test_name}")

        sys.exit(0)
    else:
        print("❌ WebSocket tests failed")
        print(f"   Tests passed: {summary['summary']['passed']}/{summary['summary']['total_tests']}")

        if summary["results"]["errors"]:
            print("\nErrors encountered:")
            for error in summary["results"]["errors"]:
                print(f"   - {error}")

        sys.exit(1)

if __name__ == "__main__":
    main()
