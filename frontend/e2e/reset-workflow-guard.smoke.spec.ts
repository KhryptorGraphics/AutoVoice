import { expect, test } from '@playwright/test'

import { mockCommonApi } from './support/mockApi'

/**
 * "Reset Workflow" clears uploaded vocals, the selected song, the resolved
 * workflow and profile overrides in a single click, with no confirmation -
 * while the same page confirms before deleting a single history record.
 *
 * The guard is deliberately conditional: confirming when there is nothing to
 * lose is friction, so the plain button has to survive too.
 */
test.describe('Reset workflow guard', () => {
  test('resets in one click when there is nothing to lose', async ({ page }) => {
    await mockCommonApi(page)
    await page.goto('/')

    await expect(page.getByTestId('reset-workflow')).toBeVisible()
    await page.getByTestId('reset-workflow').click()
    // No confirmation step: the click was the reset.
    await expect(page.getByTestId('reset-workflow-accept')).toHaveCount(0)
  })

  test('confirms once uploaded vocals would be discarded', async ({ page }) => {
    await mockCommonApi(page)
    await page.goto('/')

    await page.locator('input[type="file"][multiple]').first().setInputFiles({
      name: 'my-take.wav',
      mimeType: 'audio/wav',
      buffer: Buffer.alloc(2048),
    })

    await page.getByTestId('reset-workflow').click()
    const accept = page.getByTestId('reset-workflow-accept')
    await expect(accept).toBeVisible()
    await expect(page.getByText(/uploaded vocal file/)).toBeVisible()

    // Cancelling must leave the upload intact.
    await page.getByTestId('reset-workflow-cancel').click()
    await expect(accept).toHaveCount(0)
    await expect(page.getByTestId('reset-workflow')).toBeVisible()
  })
})
