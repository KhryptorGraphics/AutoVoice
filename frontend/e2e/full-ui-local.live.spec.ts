import { expect, test } from '@playwright/test'

import { installBrowserAudioMocks } from './support/browserAudio'
import { createWavBuffer } from './support/mockApi'

const routes = [
  { path: '/', heading: 'Voice Conversion' },
  { path: '/profiles', heading: 'Voice Profiles' },
  { path: '/youtube', heading: 'YouTube Download' },
  { path: '/diarization', heading: 'Speaker Diarization' },
  { path: '/history', heading: 'Conversion History' },
  { path: '/system', heading: 'Operator Console' },
  { path: '/karaoke', heading: 'Live Karaoke' },
  { path: '/help', heading: 'Live Karaoke Help' },
]

test.describe('Local full UI workflow coverage', () => {
  test('loads every primary route and exercises representative user functions', async ({ page }) => {
    await installBrowserAudioMocks(page)
    await page.addInitScript(() => {
      const globalAny = globalThis as typeof globalThis & {
        __AUTOVOICE_TEST_STREAMING__?: Record<string, unknown>
      }

      globalAny.__AUTOVOICE_TEST_STREAMING__ = {
        connect: () => undefined,
        startSession: ({ pipelineType, options }: { pipelineType: string; options?: { profileId?: string; collectSamples?: boolean } }) => ({
          session_id: `live-ui-${Date.now()}`,
          requested_pipeline: pipelineType,
          resolved_pipeline: pipelineType,
          runtime_backend: 'playwright-harness',
          target_profile_id: options?.profileId ?? null,
          source_voice_model_id: 'live_demo_artist',
          active_model_type: options?.profileId ? 'adapter' : 'base',
          sample_collection_enabled: Boolean(options?.collectSamples),
          audio_router_targets: {
            speaker_device: 0,
            headphone_device: 1,
          },
        }),
        startStreaming: () => undefined,
        stopStreaming: () => undefined,
        endSession: () => undefined,
        disconnect: () => undefined,
      }
    })

    const consoleErrors: string[] = []
    page.on('console', message => {
      const text = message.text()
      if (
        message.type() === 'error' &&
        !text.includes('400 (BAD REQUEST)') &&
        !text.includes('Failed to fetch')
      ) {
        consoleErrors.push(text)
      }
    })

    for (const route of routes) {
      await page.goto(route.path)
      await expect(page.getByRole('heading', { name: route.heading })).toBeVisible()
    }

    await page.goto('/profiles')
    const profileName = `Full UI ${Date.now()}`
    await page.getByTestId('create-profile-btn').click()
    await page.getByTestId('profile-name-input').fill(profileName)
    await page.getByTestId('profile-audio-input').setInputFiles({
      name: 'profile-reference.wav',
      mimeType: 'audio/wav',
      buffer: createWavBuffer(24_000, 4),
    })
    await page.locator('form').getByRole('button', { name: 'Create Profile' }).click()
    const profileCard = page.getByTestId('profile-card').filter({ hasText: profileName })
    await expect(profileCard).toBeVisible()
    await profileCard.click()
    await expect(page.getByRole('heading', { name: profileName })).toBeVisible()
    await page.locator('#quick-upload-sample').setInputFiles({
      name: 'profile-sample.wav',
      mimeType: 'audio/wav',
      buffer: createWavBuffer(24_000, 4),
    })
    await expect(page.getByText(/sample/i).first()).toBeVisible()

    await page.goto('/youtube')
    await page.locator('#youtube-url').fill('not-a-youtube-url')
    await page.getByRole('button', { name: /Fetch Info/ }).click()
    await expect(page.getByText(/error|invalid|youtube|url/i).first()).toBeVisible()

    await page.goto('/system')
    await page.getByTestId('system-offline-pipeline-select').selectOption('quality_shortcut')
    await page.getByTestId('system-live-pipeline-select').selectOption('realtime_meanvc')
    await page.getByTestId('system-runtime-save').click({ force: true })

    await page.goto('/')
    await page.locator('select').nth(1).selectOption('live-demo-singer')
    await page.locator('#artist-song-upload').setInputFiles({
      name: 'artist-song.wav',
      mimeType: 'audio/wav',
      buffer: createWavBuffer(24_000, 4),
    })
    await page.locator('#user-vocals-upload').setInputFiles({
      name: 'user-vocal.wav',
      mimeType: 'audio/wav',
      buffer: createWavBuffer(24_000, 4),
    })
    await expect(page.getByText('Ready for Conversion')).toBeVisible({ timeout: 20_000 })
    await page.getByRole('button', { name: /Convert Workflow Song/ }).click()
    await expect(page.getByText('Conversion complete', { exact: true })).toBeVisible({ timeout: 15_000 })

    await page.goto('/history')
    await expect(page.getByText('Live Demo Singer').first()).toBeVisible()
    await page.locator('input[placeholder*="Search"]').fill('Live Demo')
    await expect(page.getByText('Live Demo Singer').first()).toBeVisible()

    await page.goto('/karaoke')
    await page.getByTestId('karaoke-upload-input').setInputFiles({
      name: 'karaoke-demo.wav',
      mimeType: 'audio/wav',
      buffer: createWavBuffer(24_000, 2),
    })
    await expect(page.getByTestId('karaoke-start-button')).toBeVisible({ timeout: 15_000 })
    await expect(page.getByTestId('karaoke-speaker-device-select').locator('option')).toHaveCount(2)
    await expect(page.getByTestId('karaoke-headphone-device-select').locator('option')).toHaveCount(2)
    await page.getByTestId('karaoke-start-button').click()
    await expect(page.getByTestId('karaoke-live-indicator')).toContainText('LIVE', { timeout: 15_000 })
    await page.getByTestId('karaoke-stop-button').click()
    await expect(page.getByTestId('karaoke-start-button')).toBeVisible()

    expect(consoleErrors).toEqual([])
  })
})
