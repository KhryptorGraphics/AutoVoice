# TypeScript Style Guide

## Formatting

- **Formatter**: Prettier (default settings)
- **Linter**: ESLint with TypeScript plugin
- Semicolons: required
- Quotes: single quotes
- Indent: 2 spaces

## Naming Conventions

| Element | Convention | Example |
|---------|-----------|---------|
| Files (components) | PascalCase | `VoiceRecorder.tsx` |
| Files (utilities) | camelCase | `audioUtils.ts` |
| Interfaces | PascalCase, no `I` prefix | `VoiceProfile` |
| Types | PascalCase | `ConversionStatus` |
| Functions | camelCase | `startConversion()` |
| Constants | UPPER_SNAKE | `MAX_AUDIO_LENGTH` |
| React components | PascalCase | `AudioPlayer` |

## TypeScript Specifics

- Strict mode enabled (`"strict": true` in tsconfig)
- Prefer `interface` over `type` for object shapes
- Use `unknown` over `any` — cast explicitly when needed
- Prefer `const` assertions for literal types
- Use discriminated unions for state machines

```typescript
interface ConversionJob {
  id: string;
  status: 'pending' | 'processing' | 'complete' | 'failed';
  progress: number; // 0-100
  error?: string;
}
```

## React Patterns

- Functional components only (no class components)
- Custom hooks for shared logic (`useVoiceRecorder`, `useConversionStatus`)
- Props interfaces named `<Component>Props`
- Avoid prop drilling — use context for deep state
- Error boundaries around async operations

## API Integration

- Use typed fetch wrappers (no raw `fetch` calls)
- WebSocket messages typed with discriminated unions
- All API responses validated at boundary

```typescript
type WsMessage =
  | { type: 'progress'; jobId: string; percent: number }
  | { type: 'complete'; jobId: string; outputUrl: string }
  | { type: 'error'; jobId: string; message: string };
```

## Error Handling

- Never swallow errors silently
- Display user-friendly messages in UI
- Log technical details to console
- Use Error boundaries for component-level failures
