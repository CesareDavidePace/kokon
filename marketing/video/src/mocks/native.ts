const never = () => new Promise<never>(() => {});
export interface Connection { baseUrl: string; token: string; version: string }
export const native: Record<string, (...a: unknown[]) => Promise<never>> = new Proxy({}, { get: () => never });
