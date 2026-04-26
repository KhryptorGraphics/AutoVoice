import { expect, test } from '@playwright/test'

import { mockCommonApi } from './support/mockApi'

test.describe('Pipeline selector smoke', () => {
  test('persists the live pipeline preference and shows runtime status', async ({ page }) => {
    const mockedApi = await mockCommonApi(page)

    await page.goto('/karaoke')

    await page.getByTestId('karaoke-upload-input').setInputFiles({
      name: 'pipeline-smoke.wav',
      mimeType: 'audio/wav',
      buffer: Buffer.alloc(2048),
    })

    await expect(page.getByTestId('pipeline-option-realtime')).toBeVisible()
    await expect(page.getByTestId('pipeline-option-realtime_meanvc')).toBeVisible()
    await expect(page.getByTestId('pipeline-status-summary')).toContainText('Loaded')

    await page.getByTestId('pipeline-option-realtime_meanvc').click()

    await expect.poll(() => mockedApi.getPreferredLivePipeline()).toBe('realtime_meanvc')
    await expect(page.getByTestId('pipeline-status-summary')).toContainText('Standby')
    await expect(page.getByTestId('pipeline-status-summary')).toContainText('80 ms target')
  })
})
