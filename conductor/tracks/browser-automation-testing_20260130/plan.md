# Implementation Plan: Browser Automation Testing

**Track ID:** browser-automation-testing_20260130
**Spec:** [spec.md](./spec.md)
**Created:** 2026-01-30
**Status:** [x] MERGED into voice-profile-training_20260124
**Last Updated:** 2026-01-30

> **Note:** This track has been merged into voice-profile-training as Task 9.13.
> E2E browser testing specs remain here for reference.

## Pre-Compaction State Summary

- **BUG FIXED:** Frontend `audio` → `reference_audio` field name mismatch (api.ts:462)
- **Frontend rebuilt:** Yes, dist/ updated
- **Next action:** User retries profile upload with Connor audio file
- **Audio file:** `/home/kp/thordrive/autovoice/tests/quality_samples/conor_maynard_pillowtalk.wav`
- **Pending feature:** Voice/instrumental separation before training

## Overview

Test all AutoVoice web interface interactions using keyboard-based browser automation visible on VNC. Each phase tests a specific page/feature set.

## Phase 1: Navigation Testing (COMPLETED)

Verify all 6 pages load correctly and navigation works.

### Tasks

- [x] Task 1.1: Navigate to Convert page (/)
- [x] Task 1.2: Navigate to Karaoke page (/karaoke)
- [x] Task 1.3: Navigate to Profiles page (/profiles)
- [x] Task 1.4: Navigate to History page (/history)
- [x] Task 1.5: Navigate to System page (/system)
- [x] Task 1.6: Navigate to Help page (/help)
- [x] Task 1.7: Take screenshots of each page

### Verification

- [x] All pages load without errors
- [x] Screenshots saved to /tmp/vnc-nav-*.png

## Phase 2: System Status Verification (COMPLETED)

Verify GPU and system health displays correctly.

### Tasks

- [x] Task 2.1: Navigate to System page
- [x] Task 2.2: Verify GPU info shows "NVIDIA Thor"
- [x] Task 2.3: Verify status shows "READY"
- [x] Task 2.4: Take screenshot for documentation

### Verification

- [x] GPU Monitor displays NVIDIA Thor
- [x] Health status shows READY

## Phase 3: Voice Profiles CRUD Testing (IN PROGRESS)

Test profile creation, viewing, and deletion using Tab navigation.

### Bug Fix: Profile Upload 400 Error

**Issue:** Frontend sent `audio` field but backend expected `reference_audio`
**Fix:** Updated `frontend/src/services/api.ts` line 462 to use `reference_audio`
**Status:** FIXED - Frontend rebuilt

### Pending Requirement: Voice/Instrumental Separation

Audio samples uploaded for profile training should automatically have voice and
instrumental tracks separated, so the model trains ONLY on the voice track.
This needs verification or implementation in the backend voice cloning pipeline.

### Tasks

- [x] Task 3.1: Navigate to Profiles page
- [x] Task 3.2: Click "New Profile" button (user-assisted)
- [x] Task 3.3: Fill profile name "Connor"
- [ ] Task 3.4: Upload audio file (retry after fix)
- [ ] Task 3.5: Click Create Profile button
- [ ] Task 3.6: Verify profile appears in list
- [ ] Task 3.7: Tab to profile and open details
- [ ] Task 3.8: Navigate through tabs (Samples, Config, Jobs)
- [ ] Task 3.9: Delete test profile
- [ ] Task 3.10: Take screenshots at each step

### Verification

- [ ] Profile created successfully
- [ ] Profile appears in list with "Not Trained" badge
- [ ] Profile details viewable
- [ ] Profile deleted successfully

## Phase 4: Training Flow Testing

Test training configuration and initiation.

### Tasks

- [ ] Task 4.1: Create new profile with sample
- [ ] Task 4.2: Navigate to Config tab
- [ ] Task 4.3: Verify training config controls accessible
- [ ] Task 4.4: Tab to "Start Training" button
- [ ] Task 4.5: Verify training status changes
- [ ] Task 4.6: Check Jobs tab shows training job

### Verification

- [ ] Training can be initiated
- [ ] Status badge updates to "Training..."
- [ ] Job appears in Jobs tab

## Phase 5: Karaoke Page Testing

Test song upload and voice conversion setup.

### Tasks

- [ ] Task 5.1: Navigate to Karaoke page
- [ ] Task 5.2: Verify audio device selector displays
- [ ] Task 5.3: Test file upload via Tab + Enter
- [ ] Task 5.4: Tab to voice model dropdown
- [ ] Task 5.5: Test training sample checkbox
- [ ] Task 5.6: Test device selection dropdowns
- [ ] Task 5.7: Tab to Start Performing button

### Verification

- [ ] File upload triggers separation
- [ ] Voice model selectable
- [ ] All controls accessible via keyboard

## Phase 6: Convert Page Testing

Test basic conversion file upload.

### Tasks

- [ ] Task 6.1: Navigate to Convert page
- [ ] Task 6.2: Verify upload zone displays
- [ ] Task 6.3: Test file selection via Tab + Enter

### Verification

- [ ] Upload zone functional
- [ ] File dialog opens

## Phase 7: Error Handling Testing

Test form validation and error messages.

### Tasks

- [ ] Task 7.1: Try creating profile without name
- [ ] Task 7.2: Try creating profile without audio
- [ ] Task 7.3: Verify error messages display
- [ ] Task 7.4: Test form validation feedback

### Verification

- [ ] Error messages show for invalid input
- [ ] Form validation prevents submission

## Final Verification

- [ ] All 7 phases completed
- [ ] All acceptance criteria met
- [ ] Screenshots document each test
- [ ] No browser console errors

---

_Generated by Conductor. Tasks marked [~] in progress and [x] complete._
