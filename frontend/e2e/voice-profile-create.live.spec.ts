import { expect, test } from '@playwright/test'

import { installBrowserAudioMocks } from './support/browserAudio'
import { createWavBuffer } from './support/mockApi'

test.describe('Live backend voice profile flow', () => {
  test('creates a target profile and opens its detail view', async ({ page }) => {
    await installBrowserAudioMocks(page)
    const profileName = `Live Profile ${Date.now()}`

    await page.goto('/profiles')
    await page.getByTestId('create-profile-btn').click()
    await page.getByTestId('profile-name-input').fill(profileName)
    await page.getByTestId('profile-audio-input').setInputFiles({
      name: 'reference.wav',
      mimeType: 'audio/wav',
      buffer: createWavBuffer(),
    })

    await page.getByRole('button', { name: 'Create Profile' }).click()

    const profileCard = page.getByTestId('profile-card').filter({ hasText: profileName })
    await expect(profileCard).toBeVisible()
    await profileCard.click()

    await expect(page.getByRole('heading', { name: profileName })).toBeVisible()
  })
})
