import { expect, test } from '@playwright/test'

import { createWavBuffer, mockCommonApi } from './support/mockApi'

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

function requireTrainingPayload(payload: Record<string, unknown> | null) {
  if (!payload) {
    throw new Error('Expected training payload to be captured')
  }
  if (!isRecord(payload.config)) {
    throw new Error('Expected training payload config to be an object')
  }
  return { payload, config: payload.config }
}

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
    // Optimizer/scheduler live in the advanced section, collapsed by default.
    await page.getByTestId('training-advanced-toggle').click()
    await page.getByTestId('training-optimizer-select').selectOption('adam')
    await page.getByTestId('training-scheduler-select').selectOption('none')
    await page.getByRole('button', { name: /LoRA Fine-tune/ }).click()

    await page.getByTestId('profile-tab-samples').click()
    await expect(page.getByTestId('training-sample-selection-summary')).toContainText('2 of 2 trainable samples selected')
    await expect(page.getByTestId('training-sample-select-sample-failed')).toBeDisabled()
    await page.getByTestId('training-sample-select-sample-2').uncheck()

    await page.getByTestId('profile-tab-config').click()
    await page.getByTestId('start-training-button').click()

    await expect.poll(() => mockedApi.getLastTrainingPayload()).not.toBeNull()
    const { payload, config } = requireTrainingPayload(mockedApi.getLastTrainingPayload())
    expect(payload.sample_ids).toEqual(['sample-1'])
    expect(config.preset_id).toBe('quality_lora')
    expect(config.device_id).toBe('cuda:0')
    expect(config.precision).toBe('fp16')
    expect(config.optimizer).toBe('adam')
    expect(config.scheduler).toBe('none')
    expect(config.training_mode).toBe('lora')
    expect(config.architecture).toBe('como')
  })

  test('workflow full training sends force when clean-vocal threshold is unmet', async ({ page }) => {
    const mockedApi = await mockCommonApi(page, {
      profileOverrides: {
        full_model_eligible: false,
        full_model_remaining_minutes: 29.9,
        full_model_remaining_seconds: 1795,
      },
    })

    await page.goto('/')
    await page.locator('#artist-song-upload').setInputFiles({
      name: 'artist-song.wav',
      mimeType: 'audio/wav',
      buffer: createWavBuffer(),
    })
    await page.locator('#user-vocals-upload').setInputFiles({
      name: 'user-vocal.wav',
      mimeType: 'audio/wav',
      buffer: createWavBuffer(),
    })

    await expect(page.getByText('Training', { exact: true })).toBeVisible()
    await page.getByRole('button', { name: /Full Training/ }).click()
    await page.getByRole('button', { name: /^Start Training$/ }).click()

    await expect.poll(() => mockedApi.getLastTrainingPayload()).not.toBeNull()
    const { payload, config } = requireTrainingPayload(mockedApi.getLastTrainingPayload())
    expect(payload.force).toBe(true)
    expect(config.training_mode).toBe('full')
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
    await expect(page.getByTestId('training-output-log')).toContainText('Job started on cuda:0')
    await expect(page.getByTestId('training-output-log')).toContainText('Epoch 1/10')
  })
})
