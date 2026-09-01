import { expect, test } from '@playwright/test'

import { mockCommonApi } from './support/mockApi'

// A 390x844 iPhone-class viewport. `/history` once overflowed it by 442px with
// 502 elements past the right edge, and nothing in CI noticed.
const MOBILE_VIEWPORT = { width: 390, height: 844 }

const ROUTES = ['/', '/history', '/profiles', '/quality', '/system']

test.describe('Mobile width guard', () => {
  for (const route of ROUTES) {
    test(`${route} fits a 390px viewport`, async ({ page }) => {
      await mockCommonApi(page)
      // Viewport before navigation so responsive classes apply on first paint.
      await page.setViewportSize(MOBILE_VIEWPORT)
      await page.goto(route)
      await page.waitForLoadState('networkidle')

      const measured = await page.evaluate(() => {
        const root = document.documentElement
        const clientWidth = root.clientWidth
        const offenders = Array.from(document.querySelectorAll<HTMLElement>('body *'))
          .map((element) => ({ element, rect: element.getBoundingClientRect() }))
          .filter(({ element, rect }) => {
            if (rect.width <= 0 || rect.height <= 0) return false
            if (rect.right <= clientWidth + 1) return false
            // Off-canvas drawers sit outside on purpose and create no scroll.
            return getComputedStyle(element).position !== 'fixed'
          })
          .map(({ element, rect }) => ({
            tag: element.tagName.toLowerCase(),
            testId: element.dataset.testid ?? null,
            className: typeof element.className === 'string' ? element.className : '',
            right: Math.round(rect.right),
            width: Math.round(rect.width),
          }))
          .sort((a, b) => b.right - a.right)

        return {
          scrollWidth: root.scrollWidth,
          clientWidth,
          offenderCount: offenders.length,
          widest: offenders.slice(0, 5),
        }
      })

      expect(
        measured.scrollWidth,
        `${route} overflows by ${measured.scrollWidth - measured.clientWidth}px `
          + `(${measured.offenderCount} elements past the right edge). Widest:\n`
          + JSON.stringify(measured.widest, null, 2),
      ).toBeLessThanOrEqual(measured.clientWidth + 1)
    })
  }
})
