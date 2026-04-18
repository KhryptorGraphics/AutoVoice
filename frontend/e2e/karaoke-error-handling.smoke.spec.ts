import { expect, test } from '@playwright/test'

import { mockCommonApi } from './support/mockApi'

test.describe('Karaoke error handling smoke', () => {
  test('shows upload failures to the user', async ({ page }) => {
    await mockCommonApi(page, { karaokeUploadError: 'Upload failed in smoke test' })

    await page.goto('/karaoke')

    await page.getByTestId('karaoke-upload-input').setInputFiles({
      name: 'broken.wav',
      mimeType: 'audio/wav',
      buffer: Buffer.alloc(1024),
    })

    await expect(page.getByTestId('karaoke-upload-error')).toContainText('Upload failed in smoke test')
  })

  test('shows live-session startup failures to the user', async ({ page }) => {
    await mockCommonApi(page, { streamingStartError: 'Microphone unavailable for smoke test' })

    await page.goto('/karaoke')

    await page.getByTestId('karaoke-upload-input').setInputFiles({
      name: 'pillowtalk.wav',
      mimeType: 'audio/wav',
      buffer: Buffer.alloc(2048),
    })

    await expect(page.getByTestId('karaoke-start-button')).toBeVisible()
    await page.getByTestId('karaoke-start-button').click()

    await expect(page.getByTestId('karaoke-session-error')).toContainText('Microphone unavailable for smoke test')
  })
})
