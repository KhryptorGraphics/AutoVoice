import { expect, test } from '@playwright/test'

import { mockCommonApi } from './support/mockApi'

test.describe('Production closeout browser flows', () => {
  test('downloads YouTube audio and adds it to an existing profile', async ({ page }) => {
    const mockedApi = await mockCommonApi(page)

    await page.goto('/youtube')

    await page.getByLabel('YouTube URL input').fill('https://youtu.be/smoke-video')
    await page.getByRole('button', { name: 'Manual' }).click()
    await expect(page.getByText('Smoke Song').first()).toBeVisible()

    await page.getByRole('button', { name: 'Download Audio' }).click()
    await expect(page.getByRole('heading', { name: 'Download Complete', level: 3 })).toBeVisible()

    await page.getByLabel('Select voice profile to add audio to').selectOption('profile-1')
    await page.getByRole('button', { name: 'Add to Profile' }).click()

    await expect.poll(() => mockedApi.getYouTubeInfoRequests()).toBe(1)
    await expect.poll(() => mockedApi.getYouTubeDownloadRequests()).toBe(1)
    await expect.poll(() => mockedApi.getYouTubeAddToProfileRequests()).toBe(1)
    await expect(page.getByText(/Added "Smoke Song" as sample sample-youtube-1/)).toBeVisible()
  })

  test('runs YouTube auto ingest through review before applying profile actions', async ({ page }) => {
    const mockedApi = await mockCommonApi(page)

    await page.goto('/youtube')

    await page.getByLabel('YouTube URL input').fill('https://youtu.be/smoke-video')
    await page.getByRole('button', { name: 'Auto Ingest' }).click()
    await expect(page.getByText('Auto Ingest Ready for Review')).toBeVisible({ timeout: 5_000 })

    await expect(page.getByText('Vocals Stem')).toBeVisible()
    await expect(page.getByText('Instrumental Stem')).toBeVisible()
    await expect(page.getByText('SPEAKER_00')).toBeVisible()
    await expect(page.getByText('SPEAKER_01')).toBeVisible()
    await expect(page.getByText('Best existing match:')).toBeVisible()

    await expect(page.locator('#decision-SPEAKER_00')).toHaveValue('assign_existing')
    await expect(page.locator('#profile-SPEAKER_00')).toHaveValue('profile-1')
    await expect(page.locator('#decision-SPEAKER_01')).toHaveValue('create_new')
    await page.locator('#name-SPEAKER_01').fill('Reviewed Featured Artist')

    await expect.poll(() => mockedApi.getYouTubeIngestConfirmRequests()).toBe(0)
    await page.getByRole('button', { name: 'Confirm Reviewed Profile Actions' }).click()

    await expect.poll(() => mockedApi.getYouTubeIngestStartRequests()).toBe(1)
    await expect.poll(() => mockedApi.getYouTubeIngestPollRequests()).toBeGreaterThan(0)
    await expect.poll(() => mockedApi.getYouTubeIngestConfirmRequests()).toBe(1)
    await expect(page.getByText('Applied 2 decision(s); skipped 0.')).toBeVisible()
  })

  test('confirms checkpoint rollback and deletion from a profile', async ({ page }) => {
    const mockedApi = await mockCommonApi(page)

    await page.goto('/profiles')
    await page.getByTestId('profile-card').first().click()
    await page.getByTestId('profile-tab-checkpoints').click()

    await expect(page.getByRole('heading', { name: 'Model Checkpoints' })).toBeVisible()

    await page.getByTestId('rollback-checkpoint-checkpoint-rollback').click()
    await page.getByTestId('rollback-checkpoint-checkpoint-rollback-accept').click()
    await expect.poll(() => mockedApi.getCheckpointRollbacks()).toBe(1)

    await page.getByText('v0', { exact: true }).click()
    await page.getByTestId('delete-checkpoint-checkpoint-delete').click()
    await page.getByTestId('delete-checkpoint-checkpoint-delete-accept').click()

    await expect.poll(() => mockedApi.getCheckpointDeletes()).toBe(1)
    await expect.poll(() => mockedApi.getCheckpointCount()).toBe(2)
  })

  test('confirms notification webhook deletion from the operator console', async ({ page }) => {
    await page.addInitScript(() => {
      window.localStorage.setItem('autovoice_notifications', JSON.stringify({
        browserNotifications: false,
        soundEnabled: true,
        soundVolume: 0.5,
        webhooks: [
          {
            id: 'webhook-smoke',
            url: 'https://example.invalid/autovoice-webhook',
            name: 'Smoke Webhook',
            enabled: true,
            events: ['conversion_complete', 'training_complete'],
          },
        ],
        enabledEvents: ['conversion_complete', 'conversion_error', 'training_complete', 'training_error'],
      }))
    })
    await mockCommonApi(page)

    await page.goto('/system')

    await expect(page.getByRole('heading', { name: 'Notification Settings' })).toBeVisible()
    await expect(page.getByText('Smoke Webhook')).toBeVisible()

    await page.getByTestId('delete-webhook-webhook-smoke').click()
    await page.getByTestId('delete-webhook-webhook-smoke-accept').click()

    await expect(page.getByText('Smoke Webhook')).toHaveCount(0)
    await expect(page.getByText('Webhook deleted successfully')).toBeVisible()
  })

  test('confirms conversion history deletion from the conversion page', async ({ page }) => {
    const mockedApi = await mockCommonApi(page)

    await page.goto('/')

    await expect(page.locator('h3').filter({ hasText: 'Conversion History' })).toBeVisible()
    await expect(page.getByText('demo-vocal.wav')).toBeVisible()

    await page.getByTestId('delete-conversion-record-history-1').click()
    await page.getByTestId('delete-conversion-record-history-1-accept').click()

    await expect.poll(() => mockedApi.getConversionHistoryDeletes()).toBe(1)
    await expect.poll(() => mockedApi.getConversionRecordCount()).toBe(0)
    await expect(page.getByText('No conversion records found')).toBeVisible()
  })
})
