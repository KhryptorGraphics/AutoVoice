# Manual Testing Checklist - Loading Skeletons
## Subtask 2-1: Visual Testing and Verification

**Dev Server:** http://localhost:3006/
**Date:** 2026-02-20

---

## Pre-Test Setup

1. **Open Chrome/Firefox DevTools**
   - Press F12 or right-click → Inspect
   - Open the **Network** tab
   - Enable **Disable cache** checkbox

2. **Optional: Throttle Network Speed**
   - In Network tab, change throttle from "No throttling" to **Slow 3G**
   - This will make loading states more visible

---

## Test 1: Voice Profiles Page - Main Grid

**URL:** http://localhost:3006/profiles

### Steps:
1. Clear browser cache (Ctrl+Shift+Delete or Cmd+Shift+Delete)
2. Hard reload the page (Ctrl+Shift+R or Cmd+Shift+R)
3. **Watch the initial load**

### Expected Results:
- [ ] **Before data loads:** 3 ProfileCardSkeleton components appear in a grid
- [ ] **Skeleton structure matches:** Card shape, rounded corners, similar height to real cards
- [ ] **Smooth transition:** Skeletons fade/transition to real profile cards when data loads
- [ ] **No layout shift:** Content doesn't jump or resize when loading completes
- [ ] **No console errors:** Check Console tab for any errors or warnings

### What the skeletons should look like:
- 3 cards in a grid layout
- Each card has:
  - Circular skeleton for avatar (top)
  - Rectangular skeleton for name
  - Multiple line skeletons for metadata
  - Button-shaped skeleton at bottom
  - Pulsing/shimmer animation

---

## Test 2: Profile Detail - Samples Section

**URL:** http://localhost:3006/profiles (then click into a profile)

### Steps:
1. On the /profiles page, click on any profile card
2. **Watch the samples section** as it loads

### Expected Results:
- [ ] **Before samples load:** 3 sample item skeletons appear in a vertical list
- [ ] **Skeleton structure:** Each skeleton item has icon, filename, metadata placeholders
- [ ] **Matches real layout:** Skeleton items are same height/width as real sample list items
- [ ] **Smooth transition:** Skeletons replaced with actual sample data
- [ ] **No layout shift:** List doesn't jump when samples load
- [ ] **No console errors**

### What the skeletons should look like:
- 3 list items stacked vertically
- Each item has:
  - Small square for file icon (left)
  - Rectangular skeleton for filename
  - Smaller skeleton for metadata
  - Square skeleton for delete button (right)
  - Proper spacing between items

---

## Test 3: System Status Page - Multiple Skeletons

**URL:** http://localhost:3006/status

### Steps:
1. Navigate to the status page
2. Hard reload (Ctrl+Shift+R)
3. **Watch all sections** during initial load

### Expected Results:

#### GPU Monitor Section:
- [ ] **GPUMonitorSkeleton appears** during load
- [ ] Skeleton has multiple metric blocks
- [ ] Smooth transition to real GPU metrics
- [ ] No layout shift

#### Health Status Section:
- [ ] **Single CardSkeleton appears** during load
- [ ] Card shape matches real health status card
- [ ] Smooth transition to real health data
- [ ] No layout shift

#### Models Section:
- [ ] **3 CardSkeletons appear in a grid** during load
- [ ] Grid layout matches real model cards
- [ ] All 3 skeletons transition simultaneously to real data
- [ ] No layout shift

#### Overall:
- [ ] **No console errors or warnings**
- [ ] **All animations smooth** (no jank or flashing)
- [ ] **Professional appearance** - skeletons enhance UX

---

## Test 4: Network Throttling (Slow 3G)

**Purpose:** Make skeletons more visible and verify they handle slow loads gracefully

### Steps:
1. In DevTools → Network tab, set throttle to **Slow 3G**
2. Repeat Tests 1-3 above
3. Observe skeletons for 3-5 seconds before data loads

### Expected Results:
- [ ] Skeletons remain stable during long loading periods
- [ ] No flickering or animation glitches
- [ ] Skeletons provide clear visual feedback that something is loading
- [ ] User experience feels professional, not broken

---

## Test 5: Rapid Navigation

**Purpose:** Verify skeletons work correctly with quick page changes

### Steps:
1. Navigate: Profiles → Status → Profiles → Status (quickly)
2. Click into a profile, back out, click into another profile (quickly)

### Expected Results:
- [ ] Skeletons appear consistently on each navigation
- [ ] No "flash of empty content" before skeletons
- [ ] No leftover skeletons after data loads
- [ ] React renders cleanly without errors

---

## Test 6: Console Check

### Steps:
1. Open DevTools → Console tab
2. Reload each page (profiles, status)
3. Check for any warnings or errors

### Expected Results:
- [ ] **No React warnings** about keys, hooks, etc.
- [ ] **No errors** related to skeleton components
- [ ] **No 404s** for missing resources
- [ ] Only expected API calls (no infinite loops)

---

## Final Verification Checklist

Before marking this subtask complete, confirm:

- [ ] All skeleton components render correctly
- [ ] Layout remains stable (no CLS - Cumulative Layout Shift)
- [ ] Transitions are smooth and professional
- [ ] No console errors or warnings
- [ ] Skeletons improve perceived performance
- [ ] Code follows existing patterns (checked in previous subtasks)
- [ ] All 6 tests above pass

---

## Issues Found

**If you find any issues, document them here:**

```
Issue 1: [Description]
- Page:
- Expected:
- Actual:
- Severity: Critical/High/Medium/Low

Issue 2: ...
```

---

## Sign-Off

**Tester:** _________________
**Date:** _________________
**Status:** ☐ PASS  ☐ FAIL (with issues documented)

**Notes:**
