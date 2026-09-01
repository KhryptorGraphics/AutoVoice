import { expect, test } from '@playwright/test'

import { mockCommonApi } from './support/mockApi'

// `/help` was authored for a light theme on a dark app (text-gray-700 on
// bg-gray-900, ~1.5:1) and a `/history` filter button rendered white on white.
// Source review missed both; only computed colours caught them.
const MIN_CONTRAST = 3

const ROUTES = ['/', '/help', '/history', '/profiles', '/quality', '/system']

test.describe('Contrast guard', () => {
  for (const route of ROUTES) {
    test(`${route} renders text at >= ${MIN_CONTRAST}:1`, async ({ page }) => {
      await mockCommonApi(page)
      await page.goto(route)
      await page.waitForLoadState('networkidle')

      const failures = await page.evaluate((minContrast) => {
        type Rgba = { r: number; g: number; b: number; a: number }

        const parse = (value: string): Rgba | null => {
          const parts = value.match(/[\d.]+/g)
          if (!parts || parts.length < 3) return null
          return {
            r: Number(parts[0]),
            g: Number(parts[1]),
            b: Number(parts[2]),
            a: parts.length > 3 ? Number(parts[3]) : 1,
          }
        }

        // src-over: `top` painted onto opaque `base`.
        const composite = (top: Rgba, base: Rgba): Rgba => ({
          r: top.r * top.a + base.r * (1 - top.a),
          g: top.g * top.a + base.g * (1 - top.a),
          b: top.b * top.a + base.b * (1 - top.a),
          a: 1,
        })

        const luminance = ({ r, g, b }: Rgba): number => {
          const channel = (value: number) => {
            const c = value / 255
            return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4
          }
          return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)
        }

        const ratio = (fg: Rgba, bg: Rgba): number => {
          const [lighter, darker] = [luminance(fg), luminance(bg)].sort((a, b) => b - a)
          return (lighter + 0.05) / (darker + 0.05)
        }

        // Walk up collecting translucent layers until an opaque one is reached.
        // Stopping at the first alpha > 0 turns every `bg-*-500/10` overlay tile
        // in this app into a false failure.
        const effectiveBackground = (element: Element): Rgba | null => {
          const layers: Rgba[] = []
          let node: Element | null = element
          while (node) {
            const style = getComputedStyle(node)
            if (style.backgroundImage !== 'none') return null // gradients/images: not measurable
            const colour = parse(style.backgroundColor)
            if (colour && colour.a > 0) {
              if (colour.a >= 1) {
                return layers.reduceRight((base, layer) => composite(layer, base), colour)
              }
              layers.push(colour)
            }
            node = node.parentElement
          }
          // Nothing opaque all the way up: the browser canvas is white.
          return layers.reduceRight((base, layer) => composite(layer, base), { r: 255, g: 255, b: 255, a: 1 })
        }

        const hasOwnText = (element: Element) =>
          Array.from(element.childNodes).some(
            (node) => node.nodeType === Node.TEXT_NODE && (node.textContent ?? '').trim().length > 0,
          )

        const results: Array<{
          tag: string
          className: string
          text: string
          color: string
          background: string
          ratio: number
        }> = []

        for (const element of Array.from(document.querySelectorAll('body *'))) {
          if (!hasOwnText(element)) continue
          if (element.closest('[disabled]')) continue // WCAG exempts disabled controls
          if (element.getClientRects().length === 0) continue
          const style = getComputedStyle(element)
          if (style.visibility === 'hidden' || Number(style.opacity) === 0) continue

          const fg = parse(style.color)
          if (!fg || fg.a === 0) continue
          const bg = effectiveBackground(element)
          if (!bg) continue

          const value = ratio(composite(fg, bg), bg)
          if (value >= minContrast) continue
          results.push({
            tag: element.tagName.toLowerCase(),
            className: typeof element.className === 'string' ? element.className : '',
            text: (element.textContent ?? '').trim().slice(0, 60),
            color: style.color,
            background: `rgb(${Math.round(bg.r)}, ${Math.round(bg.g)}, ${Math.round(bg.b)})`,
            ratio: Number(value.toFixed(2)),
          })
        }
        return results
      }, MIN_CONTRAST)

      expect(failures, `${route} has text below ${MIN_CONTRAST}:1:\n${JSON.stringify(failures, null, 2)}`).toEqual([])
    })
  }
})
