import { expect, test } from '@playwright/test'

import { mockCommonApi } from './support/mockApi'

test.describe('Quality page smoke', () => {
  test('renders quality overview, selector-backed analysis, and profile drill-in', async ({ page }) => {
    const conversionRecords = [
      {
        id: 'history-1',
        status: 'complete',
        created_at: '2026-04-18T00:00:00Z',
        input_file: 'demo-vocal.wav',
        profile_id: 'profile-1',
        pipeline_type: 'quality_seedvc',
        resolved_pipeline: 'quality_seedvc',
        runtime_backend: 'tensorrt',
        adapter_type: 'hq',
        active_model_type: 'adapter',
        duration: 132,
        rtf: 0.46,
        targetVoice: 'Smoke Profile',
        originalFileName: 'demo-vocal.wav',
        quality_metrics: { quality_score: 0.91, speaker_similarity: 0.9 },
      },
      {
        id: 'history-2',
        status: 'complete',
        created_at: '2026-04-18T00:05:00Z',
        input_file: 'demo-vocal.wav',
        profile_id: 'profile-1',
        pipeline_type: 'realtime',
        resolved_pipeline: 'realtime',
        runtime_backend: 'pytorch',
        adapter_type: 'nvfp4',
        active_model_type: 'adapter',
        duration: 132,
        rtf: 0.22,
        targetVoice: 'Smoke Profile',
        originalFileName: 'demo-vocal.wav',
        quality_metrics: { quality_score: 0.84, speaker_similarity: 0.82 },
      },
    ]
    const mockedApi = await mockCommonApi(page, { conversionRecords })
    await page.goto('/quality')

    await expect(page.getByRole('heading', { name: 'Quality', exact: true })).toBeVisible()
    await expect(page.getByTestId('quality-profiles-table')).toContainText('Smoke Profile')
    await expect(page.getByTestId('quality-analyze-form')).toBeVisible()
    await expect(page.getByTestId('quality-compare-form')).toBeVisible()

    await expect(page.getByTestId('quality-model-summary')).toContainText('hq')
    await expect(page.getByTestId('quality-profile-summary')).toContainText('Smoke Profile')

    await page.getByTestId('quality-conversion-selector').selectOption('history-1')
    await page.getByRole('button', { name: 'Analyze conversion' }).click()
    await expect(page.getByTestId('quality-analyze-result')).toContainText('Quality score')
    await expect.poll(() => mockedApi.getAnalyzeRequests()).toBe(1)

    await page.getByTestId('quality-compare-source-selector').selectOption('input:demo-vocal.wav')
    await page.getByRole('checkbox').nth(0).check()
    await page.getByRole('checkbox').nth(1).check()
    await page.getByRole('button', { name: 'Compare methodologies' }).click()
    await expect(page.getByTestId('quality-compare-result')).toContainText('Best methodology')
    await expect.poll(() => mockedApi.getCompareRequests()).toBe(1)

    await page.getByTestId('quality-profiles-table').getByText('Smoke Profile').click()
    await expect(page.getByTestId('quality-profile-detail')).toBeVisible()

    await page.getByRole('button', { name: 'Check degradation' }).click()
    await expect.poll(() => mockedApi.getDegradationChecks()).toBe(1)
  })
})
