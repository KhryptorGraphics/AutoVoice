import { expect, test } from '@playwright/test'

import { mockCommonApi } from './support/mockApi'

/**
 * The training monitor reported elapsed time but no rate, so a run that kept
 * emitting metrics while running ~180x slower than normal showed a healthy
 * "running 50%" for hours. The staleness banner cannot catch that - updates
 * never stopped arriving, they just described a dying run.
 *
 * These guard both directions: the rate surface exists, and a healthy job does
 * NOT get flagged. A warning that cries wolf would be worse than no warning.
 */
test.describe('Training throughput surface', () => {
  test('live monitor exposes per-step rate and remaining estimate', async ({ page }) => {
    await mockCommonApi(page)

    await page.goto('/profiles')
    await page.getByTestId('profile-card').first().click()
    await page.getByTestId('profile-tab-jobs').click()
    await page.getByTestId('training-job-card').first().click()

    await expect(page.getByTestId('live-training-monitor')).toBeVisible()
    await expect(page.getByTestId('training-throughput')).toBeVisible()
    await expect(page.getByTestId('training-eta')).toBeVisible()
  })

  test('a healthy job is not flagged as degraded', async ({ page }) => {
    await mockCommonApi(page)

    await page.goto('/profiles')
    await page.getByTestId('profile-card').first().click()
    await page.getByTestId('profile-tab-jobs').click()
    await page.getByTestId('training-job-card').first().click()

    await expect(page.getByTestId('live-training-monitor')).toBeVisible()
    // Degradation needs a rate at least 3x worse than this run's own best AND
    // slower than the 5s floor; a steady mock satisfies neither.
    await expect(page.getByTestId('training-throughput-degraded')).toHaveCount(0)
  })
})
