import { expect, test } from '@playwright/test'

import { createWavBuffer, mockCommonApi } from './support/mockApi'

test.describe('Readiness actions', () => {
  test('profile blockers render the control that clears them', async ({ page }) => {
    await mockCommonApi(page, {
      profileOverrides: {
        has_trained_model: false,
        has_adapter_model: false,
        active_model_type: 'base',
        training_status: 'pending',
        readiness: {
          training: { ready: false, reason: 'no_trainable_samples', sample_count: 0, clean_vocal_minutes: 0 },
          conversion: { ready: false, reason: 'target_profile_not_trained' },
          live_conversion: { ready: false, reason: 'target_profile_not_trained' },
        },
      },
    })

    await page.goto('/profiles')
    await page.getByTestId('profile-card').first().click()

    await expect(page.getByTestId('profile-readiness-actions')).toContainText('no trainable samples')
    await page.getByTestId('readiness-action-add-samples').click()
    await expect(page.getByTestId('profile-tab-samples')).toHaveClass(/bg-gray-700/)

    await page.getByTestId('readiness-action-train').first().click()
    await expect(page.getByTestId('start-training-button')).toBeVisible()
  })

  test('an untrained conversion target offers training in place', async ({ page }) => {
    await mockCommonApi(page)

    await page.goto('/')
    await page.locator('#artist-song-upload').setInputFiles({
      name: 'artist-song.wav',
      mimeType: 'audio/wav',
      buffer: createWavBuffer(),
    })
    await page.locator('#user-vocals-upload').setInputFiles({
      name: 'user-vocal.wav',
      mimeType: 'audio/wav',
      buffer: createWavBuffer(),
    })

    // target_profile_not_trained becomes a button, not a sentence.
    await page.getByTestId('readiness-action-train').first().click()
    await expect(page.locator('#workflow-training')).toBeInViewport()
  })
})
