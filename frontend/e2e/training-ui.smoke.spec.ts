import { expect, test } from '@playwright/test'

import { mockCommonApi } from './support/mockApi'

test.describe('Training UI smoke', () => {
  test('submits selected samples and granular backend-supported config', async ({ page }) => {
    const mockedApi = await mockCommonApi(page)

    await page.goto('/profiles')

    await page.getByTestId('profile-card').first().click()
    await expect(page.getByTestId('training-readiness-panel')).toBeVisible()
    await page.getByTestId('profile-tab-config').click()

    await page.getByTestId('training-preset-selector').locator('select').selectOption('quality_lora')
    await page.getByTestId('training-device-select').selectOption('cuda:0')
    await page.getByTestId('training-precision-select').selectOption('fp16')
    await page.getByTestId('training-optimizer-select').selectOption('adam')
    await page.getByTestId('training-scheduler-select').selectOption('none')

    await page.getByTestId('profile-tab-samples').click()
    await expect(page.getByTestId('training-sample-selection-summary')).toContainText('2 of 2 trainable samples selected')
    await expect(page.getByTestId('training-sample-select-sample-failed')).toBeDisabled()
    await page.getByTestId('training-sample-select-sample-2').uncheck()

    await page.getByTestId('profile-tab-config').click()
    await page.getByTestId('start-training-button').click()

    await expect.poll(() => mockedApi.getLastTrainingPayload()).not.toBeNull()
    const payload = mockedApi.getLastTrainingPayload() as {
      sample_ids: string[]
      config: Record<string, unknown>
    }
    expect(payload.sample_ids).toEqual(['sample-1'])
    expect(payload.config.preset_id).toBe('quality_lora')
    expect(payload.config.device_id).toBe('cuda:0')
    expect(payload.config.precision).toBe('fp16')
    expect(payload.config.optimizer).toBe('adam')
    expect(payload.config.scheduler).toBe('none')
  })

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
