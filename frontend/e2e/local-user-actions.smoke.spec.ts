import { expect, test } from '@playwright/test'

import { createWavBuffer, mockCommonApi } from './support/mockApi'

const primaryRoutes = [
  { path: '/', heading: 'Voice Conversion' },
  { path: '/karaoke', heading: 'Live Karaoke' },
  { path: '/singalong', heading: 'Sing-Along Recording Studio' },
  { path: '/profiles', heading: 'Voice Profiles' },
  { path: '/samples', heading: 'Training Sample Inbox' },
  { path: '/youtube', heading: 'YouTube Download' },
  { path: '/diarization', heading: 'Speaker Diarization' },
  { path: '/history', heading: 'Conversion History' },
  { path: '/system', heading: 'Operator Console' },
  { path: '/help', heading: 'Live Karaoke Help' },
]

test.describe('Local user action coverage', () => {
  test('navigates every primary route and exercises local-only secondary actions', async ({ page }) => {
    const mockedApi = await mockCommonApi(page, { apiToken: 'operator-smoke-token' })

    for (const route of primaryRoutes) {
      await page.goto(route.path)
      await expect(page.getByRole('heading', { name: route.heading })).toBeVisible()
    }

    await page.goto('/samples')
    await expect(page.getByTestId('sample-inbox-page')).toBeVisible()
    await expect(page.getByText('youtube_ingest_review')).toBeVisible()
    await expect(page.getByText('browser_singalong_capture')).toBeVisible()
    await page.getByTestId('sample-inbox-filter').selectOption('trainable')
    await expect(page.getByText('youtube_ingest_review')).toBeVisible()
    await expect(page.getByText('browser_singalong_capture')).toHaveCount(0)
    await page.getByTestId('sample-inbox-filter').selectOption('blocked')
    await expect(page.getByText('browser_singalong_capture')).toBeVisible()
    await expect.poll(() => mockedApi.getSampleReviewRequests()).toBeGreaterThanOrEqual(3)

    await page.goto('/diarization')
    await page.locator('input[type="file"]').setInputFiles({
      name: 'Smoke Artist ft. Guest.wav',
      mimeType: 'audio/wav',
      buffer: createWavBuffer(24_000, 3),
    })
    await page.getByRole('button', { name: 'Run Diarization' }).click()
    await expect(page.getByRole('heading', { name: 'Results Summary' })).toBeVisible()
    await expect(page.getByText('SPEAKER_00').first()).toBeVisible()
    await page.getByRole('button', { name: '+ Guest' }).click()
    await page.getByRole('button', { name: 'Create', exact: true }).click()
    await expect.poll(() => mockedApi.getDiarizationProfileCreates()).toBe(1)
    await page.locator('select').filter({ hasText: 'Smoke Profile' }).first().selectOption('profile-1')
    await page.getByRole('button', { name: 'Confirm Assignment' }).click()
    await expect.poll(() => mockedApi.getDiarizationRequests()).toBe(1)
    await expect.poll(() => mockedApi.getDiarizationAssignments()).toBe(1)

    await page.goto('/system')
    await expect(page.getByTestId('local-production-wizard')).toContainText('Local ready')
    await page.getByRole('button', { name: 'Export Backup' }).click()
    await expect(page.getByText(/Exported 2 files/)).toBeVisible()
    await page.locator('input[type="file"][accept*=".zip"]').setInputFiles({
      name: 'autovoice-smoke.zip',
      mimeType: 'application/zip',
      buffer: Buffer.from('PK\x03\x04autovoice-smoke'),
    })
    await expect(page.getByText('autovoice-smoke.zip', { exact: true })).toBeVisible()
    await page.getByRole('button', { name: 'Dry Run' }).click()
    await expect(page.getByText(/Dry-run read 1 files/)).toBeVisible()
    await page.getByRole('button', { name: 'Apply Restore' }).click()
    await expect(page.getByText(/Restored 1 files/)).toBeVisible()
    await expect.poll(() => mockedApi.getBackupExports()).toBe(1)
    await expect.poll(() => mockedApi.getBackupDryRuns()).toBe(1)
    await expect.poll(() => mockedApi.getBackupApplies()).toBe(1)

    await page.goto('/help')
    await expect(page.getByText('Quick Start')).toBeVisible()
    await expect(page.getByText('Browser Sing-Along Recording')).toBeVisible()
    await expect(page.getByRole('link', { name: '/api/v1/karaoke/health' })).toBeVisible()
  })
})
