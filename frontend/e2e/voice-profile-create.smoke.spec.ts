import { expect, test } from '@playwright/test'

import { mockCommonApi } from './support/mockApi'

test.describe('Voice profile creation smoke', () => {
  test('creates a profile from uploaded reference audio', async ({ page }) => {
    const mockedApi = await mockCommonApi(page)

    await page.goto('/profiles')

    await page.getByTestId('create-profile-btn').click()
    await page.getByTestId('profile-name-input').fill('Created Smoke Profile')
    await page.getByTestId('profile-audio-input').setInputFiles({
      name: 'reference.wav',
      mimeType: 'audio/wav',
      buffer: Buffer.alloc(1024),
    })

    await page.getByRole('button', { name: 'Create Profile' }).click()

    await expect.poll(() => mockedApi.getProfileCount()).toBe(2)
    await expect(page.getByTestId('profile-card').first()).toContainText('Created Smoke Profile')
  })
})
