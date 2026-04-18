import { expect, test } from '@playwright/test'

import { mockCommonApi } from './support/mockApi'

test.describe('Training UI smoke', () => {
  test('renders live controls, preview, and pause/resume flow', async ({ page }) => {
    const mockedApi = await mockCommonApi(page)

    await page.goto('/profiles')

    await page.getByTestId('profile-card').first().click()
    await page.getByTestId('profile-tab-jobs').click()
    await page.getByTestId('training-job-card').first().click()

    await expect(page.getByTestId('live-training-monitor')).toBeVisible()
    await expect(page.getByTestId('training-checkpoint-path')).toContainText('checkpoint_step_1000')

    await page.getByTestId('pause-training-button').click()
    await expect.poll(() => mockedApi.isPaused()).toBe(true)

    await page.getByTestId('resume-training-button').click()
    await expect.poll(() => mockedApi.isPaused()).toBe(false)

    await page.getByTestId('generate-training-preview').click()
    await expect(page.getByTestId('training-preview-audio')).toBeVisible()
  })
})
