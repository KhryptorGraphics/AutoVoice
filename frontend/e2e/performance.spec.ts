import { test, expect } from '@playwright/test'

test.describe('Performance Tests', () => {
  test.describe('History List Performance', () => {
    test('should render 100 items without performance degradation', async ({ page }) => {
      // Mock API to return 100 items
      await page.route('**/api/v1/history*', (route) => {
        const items = Array.from({ length: 100 }, (_, i) => ({
          id: `item-${i}`,
          filename: `song-${i}.mp3`,
          status: i % 3 === 0 ? 'completed' : i % 3 === 1 ? 'processing' : 'failed',
          created_at: new Date(Date.now() - i * 3600000).toISOString(),
          duration: Math.floor(Math.random() * 300) + 60,
          profile_name: `Profile ${i % 5}`,
        }))
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ items, total: 100, page: 1, per_page: 100 }),
        })
      })

      // Measure initial load time
      const startTime = Date.now()
      await page.goto('/history')

      // Wait for items to render
      const historyItems = page.locator('[data-testid="history-item"]')
      await expect(historyItems.first()).toBeVisible()
      const loadTime = Date.now() - startTime

      // Should load within 3 seconds
      expect(loadTime).toBeLessThan(3000)

      // Verify all items rendered
      await expect(historyItems).toHaveCount(100)
    })

    test('should scroll smoothly through 100 items', async ({ page }) => {
      await page.route('**/api/v1/history*', (route) => {
        const items = Array.from({ length: 100 }, (_, i) => ({
          id: `item-${i}`,
          filename: `song-${i}.mp3`,
          status: 'completed',
          created_at: new Date(Date.now() - i * 3600000).toISOString(),
        }))
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ items, total: 100 }),
        })
      })

      await page.goto('/history')

      // Wait for content
      await page.locator('[data-testid="history-item"]').first().waitFor()

      // Measure scroll performance
      const scrollMetrics = await page.evaluate(async () => {
        const container = document.querySelector('[data-testid="history-list"]') as HTMLElement
        if (!container) return { frames: 0, avgFrameTime: 0 }

        const frameTimings: number[] = []
        let lastTime = performance.now()

        return new Promise<{ frames: number; avgFrameTime: number }>((resolve) => {
          const measureFrame = () => {
            const now = performance.now()
            frameTimings.push(now - lastTime)
            lastTime = now
          }

          const scrollListener = () => measureFrame()
          container.addEventListener('scroll', scrollListener)

          // Scroll to bottom
          container.scrollTo({ top: container.scrollHeight, behavior: 'smooth' })

          setTimeout(() => {
            container.removeEventListener('scroll', scrollListener)
            const avgFrameTime =
              frameTimings.length > 0
                ? frameTimings.reduce((a, b) => a + b, 0) / frameTimings.length
                : 0
            resolve({ frames: frameTimings.length, avgFrameTime })
          }, 2000)
        })
      })

      // Average frame time should be under 32ms (30+ fps)
      if (scrollMetrics.frames > 0) {
        expect(scrollMetrics.avgFrameTime).toBeLessThan(32)
      }
    })

    test('should filter 100 items quickly', async ({ page }) => {
      await page.route('**/api/v1/history*', (route) => {
        const url = new URL(route.request().url())
        const status = url.searchParams.get('status')

        const items = Array.from({ length: 100 }, (_, i) => ({
          id: `item-${i}`,
          filename: `song-${i}.mp3`,
          status: i % 3 === 0 ? 'completed' : i % 3 === 1 ? 'processing' : 'failed',
          created_at: new Date(Date.now() - i * 3600000).toISOString(),
        })).filter((item) => !status || item.status === status)

        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ items, total: items.length }),
        })
      })

      await page.goto('/history')
      await page.locator('[data-testid="history-item"]').first().waitFor()

      // Apply filter and measure
      const startTime = Date.now()
      await page.locator('[data-testid="status-filter"]').selectOption('completed')

      // Wait for filtered results
      await page.waitForFunction(() => {
        const items = document.querySelectorAll('[data-testid="history-item"]')
        return items.length < 100
      })

      const filterTime = Date.now() - startTime

      // Filter should complete within 500ms
      expect(filterTime).toBeLessThan(500)
    })
  })

  test.describe('Conversion Page Performance', () => {
    test('should load conversion page within performance budget', async ({ page }) => {
      const metrics = await page.evaluate(() => {
        return new Promise<PerformanceNavigationTiming>((resolve) => {
          const observer = new PerformanceObserver((list) => {
            const entries = list.getEntries()
            const navigation = entries.find(
              (e) => e.entryType === 'navigation'
            ) as PerformanceNavigationTiming
            if (navigation) resolve(navigation)
          })
          observer.observe({ entryTypes: ['navigation'] })
        })
      })

      await page.goto('/convert')

      // Get Core Web Vitals
      const webVitals = await page.evaluate(() => {
        return new Promise<{ lcp: number; fid: number; cls: number }>((resolve) => {
          let lcp = 0,
            fid = 0,
            cls = 0

          new PerformanceObserver((list) => {
            const entries = list.getEntries()
            lcp = entries[entries.length - 1]?.startTime || 0
          }).observe({ entryTypes: ['largest-contentful-paint'] })

          new PerformanceObserver((list) => {
            const entries = list.getEntries() as PerformanceEventTiming[]
            fid = entries[0]?.processingStart - entries[0]?.startTime || 0
          }).observe({ entryTypes: ['first-input'] })

          new PerformanceObserver((list) => {
            for (const entry of list.getEntries() as any[]) {
              if (!entry.hadRecentInput) {
                cls += entry.value
              }
            }
          }).observe({ entryTypes: ['layout-shift'] })

          setTimeout(() => resolve({ lcp, fid, cls }), 3000)
        })
      })

      // LCP should be under 2.5s
      expect(webVitals.lcp).toBeLessThan(2500)

      // CLS should be under 0.1
      expect(webVitals.cls).toBeLessThan(0.1)
    })

    test('should handle rapid setting changes without lag', async ({ page }) => {
      await page.goto('/convert')

      const pitchSlider = page.locator('[data-testid="pitch-shift-slider"]')
      await pitchSlider.waitFor()

      // Rapid slider changes
      const startTime = Date.now()
      for (let i = -12; i <= 12; i++) {
        await pitchSlider.fill(String(i))
      }
      const elapsed = Date.now() - startTime

      // 24 changes should complete within 2 seconds
      expect(elapsed).toBeLessThan(2000)
    })
  })

  test.describe('GPU Monitor Performance', () => {
    test('should update GPU metrics without memory leaks', async ({ page }) => {
      await page.goto('/system')

      // Get initial memory
      const initialMemory = await page.evaluate(() => {
        return (performance as any).memory?.usedJSHeapSize || 0
      })

      // Wait for multiple metric updates (30 seconds)
      await page.waitForTimeout(30000)

      // Get final memory
      const finalMemory = await page.evaluate(() => {
        return (performance as any).memory?.usedJSHeapSize || 0
      })

      // Memory growth should be less than 10MB
      const memoryGrowth = finalMemory - initialMemory
      expect(memoryGrowth).toBeLessThan(10 * 1024 * 1024)
    })

    test('should render GPU utilization chart efficiently', async ({ page }) => {
      await page.goto('/system')

      const chart = page.locator('[data-testid="gpu-utilization-chart"]')
      await chart.waitFor()

      // Measure chart update performance
      const chartPerformance = await page.evaluate(() => {
        const chart = document.querySelector('[data-testid="gpu-utilization-chart"]')
        if (!chart) return { renderCount: 0, avgRenderTime: 0 }

        let renderCount = 0
        let totalRenderTime = 0

        const observer = new MutationObserver(() => {
          const start = performance.now()
          // Force a layout calculation
          chart.getBoundingClientRect()
          totalRenderTime += performance.now() - start
          renderCount++
        })

        observer.observe(chart, { childList: true, subtree: true })

        return new Promise<{ renderCount: number; avgRenderTime: number }>((resolve) => {
          setTimeout(() => {
            observer.disconnect()
            resolve({
              renderCount,
              avgRenderTime: renderCount > 0 ? totalRenderTime / renderCount : 0,
            })
          }, 10000)
        })
      })

      // Average render time should be under 16ms (60fps)
      if (chartPerformance.renderCount > 0) {
        expect(chartPerformance.avgRenderTime).toBeLessThan(16)
      }
    })
  })
})
