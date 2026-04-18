import { expect, test } from '@playwright/test'

import { mockCommonApi } from './support/mockApi'

test.describe('Pipeline selector smoke', () => {
  test('persists the canonical pipeline preference and shows runtime status', async ({ page }) => {
    const mockedApi = await mockCommonApi(page)

    await page.goto('/')

    await expect(page.getByTestId('pipeline-option-quality')).toBeVisible()
    await expect(page.getByTestId('pipeline-status-summary')).toContainText('Standby')

    await page.getByTestId('pipeline-option-realtime').click()

    await expect.poll(() => mockedApi.getPreferredPipeline()).toBe('realtime')
    await expect(page.getByTestId('pipeline-status-summary')).toContainText('Loaded')
    await expect(page.getByTestId('pipeline-status-summary')).toContainText('100 ms target')
  })
})
