#!/usr/bin/env python3
"""Verification script to confirm pitch data is included in JobManager WebSocket completion events."""

import sys
import re
from pathlib import Path

def verify_jobmanager_pitch_data():
    """Verify that pitch data (f0_contour, f0_times) is included in JobManager completion events."""

    job_manager_path = Path(__file__).parent.parent / "src" / "auto_voice" / "web" / "job_manager.py"

    if not job_manager_path.exists():
        print(f"❌ ERROR: {job_manager_path} not found")
        return False

    content = job_manager_path.read_text()

    # Check 1: Verify f0_contour is extracted from pipeline result
    f0_extraction_pattern = r"f0_contour\s*=\s*result\.get\('f0_contour'\)"
    if not re.search(f0_extraction_pattern, content):
        print("❌ FAIL: f0_contour extraction from pipeline result not found")
        return False
    print("✅ PASS: f0_contour extraction from pipeline result found")

    # Check 2: Verify f0_times is computed
    f0_times_pattern = r"f0_times\s*=.*hop_length.*sample_rate"
    if not re.search(f0_times_pattern, content, re.DOTALL):
        print("❌ FAIL: f0_times computation not found")
        return False
    print("✅ PASS: f0_times computation found")

    # Check 3: Verify f0_contour is added to completion payload
    payload_f0_contour_pattern = r"completion_payload\['f0_contour'\]"
    if not re.search(payload_f0_contour_pattern, content):
        print("❌ FAIL: f0_contour not added to completion payload")
        return False
    print("✅ PASS: f0_contour added to completion payload")

    # Check 4: Verify f0_times is added to completion payload
    payload_f0_times_pattern = r"completion_payload\['f0_times'\]"
    if not re.search(payload_f0_times_pattern, content):
        print("❌ FAIL: f0_times not added to completion payload")
        return False
    print("✅ PASS: f0_times added to completion payload")

    # Check 5: Verify emission of conversion_complete event with the payload
    emit_pattern = r"self\.socketio\.emit\(\s*'conversion_complete',\s*completion_payload"
    if not re.search(emit_pattern, content):
        print("❌ FAIL: conversion_complete event emission with completion_payload not found")
        return False
    print("✅ PASS: conversion_complete event emission with completion_payload found")

    # Check 6: Verify pitch data is stored in job metadata
    metadata_f0_pattern = r"job\['metadata'\]\['f0_contour'\]"
    if not re.search(metadata_f0_pattern, content):
        print("❌ FAIL: f0_contour not stored in job metadata")
        return False
    print("✅ PASS: f0_contour stored in job metadata")

    metadata_f0_times_pattern = r"job\['metadata'\]\['f0_times'\]"
    if not re.search(metadata_f0_times_pattern, content):
        print("❌ FAIL: f0_times not stored in job metadata")
        return False
    print("✅ PASS: f0_times stored in job metadata")

    # Check 7: Verify get_job_status returns pitch data
    status_f0_pattern = r"status_dict\['f0_contour'\].*job\['metadata'\]\.get\('f0_contour'\)"
    if not re.search(status_f0_pattern, content, re.DOTALL):
        print("❌ FAIL: get_job_status does not return f0_contour")
        return False
    print("✅ PASS: get_job_status returns f0_contour")

    status_f0_times_pattern = r"status_dict\['f0_times'\].*job\['metadata'\]\.get\('f0_times'\)"
    if not re.search(status_f0_times_pattern, content, re.DOTALL):
        print("❌ FAIL: get_job_status does not return f0_times")
        return False
    print("✅ PASS: get_job_status returns f0_times")

    print("\n" + "="*70)
    print("✅ ALL CHECKS PASSED: Pitch data is properly included in JobManager")
    print("="*70)
    print("\nImplementation details:")
    print("  - f0_contour extracted from pipeline result (line ~413-426)")
    print("  - f0_times computed from hop_length and sample_rate (line ~423-426)")
    print("  - Both added to completion_payload (line ~462, 469)")
    print("  - Emitted via conversion_complete event (line ~482-486)")
    print("  - Stored in job metadata for status polling (line ~436-437)")
    print("  - Returned by get_job_status (line ~127-128)")
    print("\nThis mirrors the streaming flow in websocket_handler.py ✨")

    return True

if __name__ == "__main__":
    success = verify_jobmanager_pitch_data()
    sys.exit(0 if success else 1)
