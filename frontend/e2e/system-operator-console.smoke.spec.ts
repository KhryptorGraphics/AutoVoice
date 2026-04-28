import { expect, test } from '@playwright/test'

import { mockCommonApi } from './support/mockApi'

test.describe('System operator console smoke', () => {
  test('updates runtime defaults and performs model plus TensorRT operator actions', async ({ page }) => {
    const mockedApi = await mockCommonApi(page, { apiToken: 'operator-smoke-token' })

    await page.goto('/system')

    await expect(page.getByRole('heading', { name: 'Operator Console' })).toBeVisible()
    await expect(page.getByTestId('system-offline-pipeline-select')).toBeVisible()
    await expect(page.getByTestId('benchmark-evidence-panel')).toContainText('Evidence missing or blocked')
    await expect(page.getByTestId('evidence-lane-list')).toContainText('jetson-cuda-tensorrt')
    await expect(page.getByTestId('evidence-artifact-list')).toContainText('release_decision.json')
    await expect(page.getByTestId('evidence-next-actions')).toContainText('--execute on Jetson')
    await expect(page.getByText('Model Manager')).toBeVisible()
    await expect(page.getByText('TensorRT Optimization')).toBeVisible()
    await expect.poll(() => mockedApi.getAuthorizationHeaders().some((header) => header === 'Bearer operator-smoke-token')).toBe(true)

    await page.getByTestId('system-offline-pipeline-select').selectOption('quality_shortcut')
    await page.getByTestId('system-live-pipeline-select').selectOption('realtime_meanvc')
    await page.getByTestId('system-runtime-save').scrollIntoViewIfNeeded()
    await page.getByTestId('system-runtime-save').click({ force: true })

    await expect.poll(() => mockedApi.getPreferredPipeline()).toBe('quality_shortcut')
    await expect.poll(() => mockedApi.getPreferredLivePipeline()).toBe('realtime_meanvc')

    await page.getByTestId('unload-model-encoder').click()
    await expect(page.getByTestId('unload-model-encoder-accept')).toBeVisible()
    await page.getByTestId('unload-model-encoder-accept').click()
    await expect.poll(() => mockedApi.getLoadedModelTypes().includes('encoder')).toBe(false)

    await page.getByRole('button', { name: /Build TensorRT Engines/ }).click()
    await expect.poll(() => mockedApi.getBuiltTensorRTEngines()).toContain('vocoder:fp16')
  })
})
