import { mergeConfig, defineConfig } from 'vitest/config'
import viteConfig from './vite.config.ts'

// Vitest reuses the Vite config (Preact preset, /api dev proxy) and adds the
// jsdom test environment. Tests live next to their components as
// `*.test.{ts,tsx}` files and run fully offline.
export default mergeConfig(
  viteConfig,
  defineConfig({
    test: {
      environment: 'jsdom',
      globals: true,
      setupFiles: ['src/test-setup.ts'],
      include: ['src/**/*.test.{ts,tsx}'],
      css: false,
    },
  }),
)
