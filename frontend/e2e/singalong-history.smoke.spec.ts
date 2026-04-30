import { expect, test } from '@playwright/test'

import { mockCommonApi } from './support/mockApi'

test.describe('Sing-along audio I/O and history smoke', () => {
  test('exposes browser original playback, output selection, mic selection, and recording controls', async ({ page }) => {
    const mockedApi = await mockCommonApi(page)

    await page.goto('/singalong')
    await expect(page.getByTestId('singalong-page')).toBeVisible()
    await expect(page.getByTestId('singalong-empty-state')).toBeVisible()

    await page.getByTestId('singalong-original-file-input').setInputFiles({
      name: 'artist-original.wav',
      mimeType: 'audio/wav',
      buffer: Buffer.alloc(4096),
    })

    await expect(page.getByTestId('browser-singalong-capture')).toBeVisible()
    await expect(page.getByTestId('browser-capture-source-label')).toContainText('artist-original.wav')
    await expect(page.getByTestId('browser-capture-input-select').locator('option')).toHaveCount(2)
    await expect(page.getByTestId('browser-capture-output-select').locator('option')).toHaveCount(2)

    await page.getByTestId('browser-capture-enable-mic').click()
    await expect(page.getByText('Browser microphone is ready.')).toBeVisible()
    await page.getByTestId('browser-capture-output-select').selectOption('browser-output-1')
    await page.getByTestId('browser-capture-start').click()
    await expect(page.getByTestId('browser-capture-recording-status')).toBeVisible()
    await page.waitForTimeout(1100)
    await page.getByTestId('browser-capture-stop').click()
    await expect(page.getByTestId('browser-capture-take-preview')).toBeVisible()
    await expect(page.getByTestId('browser-capture-quality-status')).toContainText('warn')

    await page.getByTestId('browser-capture-attach').click()
    await expect(page.getByText('Recorded take attached to the selected profile.')).toBeVisible()
    await expect.poll(() => mockedApi.getProfileSampleCount()).toBe(4)
  })

  test('renders history when records are missing optional display fields', async ({ page }) => {
    await mockCommonApi(page, {
      conversionRecords: [
        {
          id: 'history-sparse',
          status: 'complete',
          created_at: 'not-a-date',
          input_file: '',
          profile_id: 'profile-1',
          preset: '',
        },
      ],
    })

    await page.goto('/history')
    await expect(page.getByTestId('history-page')).toBeVisible()
    await expect(page.getByText('history-sparse')).toBeVisible()
    await expect(page.getByText('Unknown date')).toBeVisible()

    await page.getByPlaceholder('Search by filename, voice, or notes...').fill('profile-1')
    await expect(page.getByText('history-sparse')).toBeVisible()
  })
})
