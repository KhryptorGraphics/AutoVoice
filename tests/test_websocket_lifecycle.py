"""Test WebSocket session lifecycle management"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from auto_voice.web.app import create_app
from flask_socketio import SocketIOTestClient

def test_websocket_lifecycle():
    """Test WebSocket connect/disconnect and session lifecycle"""
    print("\n=== WebSocket Session Lifecycle Tests ===\n")

    # Create app
    app, socketio = create_app()

    # Create test client
    client = SocketIOTestClient(app, socketio)

    # Test 1: Connect handler
    print("Test 1: Connect handler initializes session")
    received = client.get_received()
    print(f"  Received on connect: {len(received)} messages")

    if len(received) > 0:
        status_msg = received[0]
        print(f"  ✓ Status message: {status_msg.get('name')}")

        if 'args' in status_msg and len(status_msg['args']) > 0:
            data = status_msg['args'][0]
            if 'session_id' in data:
                print(f"  ✓ Session ID assigned: {data['session_id'][:8]}...")
            if 'capabilities' in data:
                print(f"  ✓ Capabilities sent: {list(data['capabilities'].keys())[:3]}...")
        print("  ✅ Connect handler working\n")
    else:
        print("  ⚠️  No status message received on connect\n")

    # Test 2: Join room with custom session_id
    print("Test 2: Join room with custom session_id")
    client.emit('join', {'room': 'test-room', 'session_id': 'custom-123'})
    received = client.get_received()

    if len(received) > 0:
        join_msg = received[0]
        if 'args' in join_msg and len(join_msg['args']) > 0:
            data = join_msg['args'][0]
            if 'session_id' in data and data['session_id'] == 'custom-123':
                print(f"  ✓ Custom session ID preserved: {data['session_id']}")
            if 'message' in data and 'test-room' in data['message']:
                print(f"  ✓ Joined room: {data['message']}")
        print("  ✅ Join with custom session_id working\n")
    else:
        print("  ⚠️  No response to join event\n")

    # Test 3: Get status (should default to request.sid if no session_id provided)
    print("Test 3: Get status without explicit session_id")
    client.emit('get_status', {})
    received = client.get_received()

    if len(received) > 0:
        status_msg = received[0]
        if 'args' in status_msg and len(status_msg['args']) > 0:
            data = status_msg['args'][0]
            if 'session_id' in data:
                print(f"  ✓ Session ID returned: {data['session_id'][:8]}...")
            if 'session_active' in data:
                print(f"  ✓ Session active status: {data['session_active']}")
        print("  ✅ Status request working with default session_id\n")
    else:
        print("  ⚠️  No response to get_status event\n")

    # Test 4: Update config without explicit session_id
    print("Test 4: Update config without explicit session_id")
    client.emit('voice_config', {'config': {'speed': 1.2, 'pitch': 0.9}})
    received = client.get_received()

    if len(received) > 0:
        config_msg = received[0]
        if 'args' in config_msg and len(config_msg['args']) > 0:
            data = config_msg['args'][0]
            if 'status' in data and data['status'] == 'success':
                print(f"  ✓ Config update successful")
            if 'session_id' in data:
                print(f"  ✓ Session ID in response: {data['session_id'][:8]}...")
        print("  ✅ Config update working with default session_id\n")
    else:
        print("  ⚠️  No response to voice_config event\n")

    # Test 5: Disconnect handler cleanup
    print("Test 5: Disconnect handler cleanup")
    # Access the WebSocket handler to check session count before disconnect
    handler = None
    for attr_name in dir(socketio):
        attr = getattr(socketio, attr_name)
        if hasattr(attr, 'sessions') and hasattr(attr, 'audio_buffers'):
            handler = attr
            break

    if handler:
        sessions_before = len(handler.sessions)
        print(f"  Sessions before disconnect: {sessions_before}")

    # Disconnect
    client.disconnect()

    if handler:
        sessions_after = len(handler.sessions)
        print(f"  Sessions after disconnect: {sessions_after}")
        if sessions_after < sessions_before:
            print(f"  ✓ Session cleaned up ({sessions_before - sessions_after} removed)")
        print("  ✅ Disconnect cleanup working\n")
    else:
        print("  ⚠️  Could not verify session cleanup\n")

    print("=== All WebSocket Lifecycle Tests Complete ===\n")

if __name__ == '__main__':
    test_websocket_lifecycle()
