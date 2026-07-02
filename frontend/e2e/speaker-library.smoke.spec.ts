import { expect, test } from '@playwright/test'

import { mockCommonApi } from './support/mockApi'

test.describe('Speaker library', () => {
  test('renders the speaker library section and runs identification', async ({ page }) => {
    const mockedApi = await mockCommonApi(page)

    await page.goto('/diarization')

    // Section and its panels render with mocked data
    await expect(page.getByTestId('speaker-library-section')).toBeVisible()
    await expect(page.getByRole('heading', { name: 'Speaker Library' })).toBeVisible()
    await expect(page.getByText('Lead Vocalist')).toBeVisible()
    await expect(page.getByText('Featured Guest').first()).toBeVisible()
    await expect(page.getByText('Smoke Song').first()).toBeVisible()

    // One interaction: run identification and confirm the API was hit
    await page.getByRole('button', { name: 'Run Identification' }).click()
    await expect.poll(() => mockedApi.getIdentifyRuns()).toBe(1)
  })
})
