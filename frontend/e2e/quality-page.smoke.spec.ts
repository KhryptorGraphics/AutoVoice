import { expect, test } from '@playwright/test'

import { mockCommonApi } from './support/mockApi'

test.describe('Quality page smoke', () => {
  test('renders quality overview and profile drill-in with degradation check', async ({ page }) => {
    const mockedApi = await mockCommonApi(page)

    await page.goto('/quality')

    await expect(page.getByRole('heading', { name: 'Quality', exact: true })).toBeVisible()
    await expect(page.getByTestId('quality-profiles-table')).toContainText('Smoke Profile')
    await expect(page.getByTestId('quality-analyze-form')).toBeVisible()
    await expect(page.getByTestId('quality-compare-form')).toBeVisible()

    await page.getByTestId('quality-profiles-table').getByText('Smoke Profile').click()
    await expect(page.getByTestId('quality-profile-detail')).toBeVisible()

    await page.getByRole('button', { name: 'Check degradation' }).click()
    await expect.poll(() => mockedApi.getDegradationChecks()).toBe(1)
  })
})
