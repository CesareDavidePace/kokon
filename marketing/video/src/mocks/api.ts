// The service API, offline: image URLs point at the video's static files
// (public/img/<id>.jpg and public/thumb/<id>.jpg); requests never resolve.
import { staticFile } from "remotion";
import type { AppErrorBody } from "../../../../src/lib/types";

export class ApiError extends Error {
  constructor(public body: AppErrorBody, public status: number) {
    super(body.message);
  }
}
export const setPrivateInputs = (_on: boolean) => {};
export const configure = (_b: string, _t: string) => {};
export const isConfigured = () => true;
const never = () => new Promise<never>(() => {});
export const api: Record<string, (...a: unknown[]) => Promise<never>> = new Proxy({}, { get: () => never });
export const urls = {
  input: (id: string) => staticFile(`img/${id}.jpg`),
  inputThumb: (id: string) => staticFile(`thumb/${id}.jpg`),
  gen: (id: string) => staticFile(`img/${id}.jpg`),
  genThumb: (id: string) => staticFile(`thumb/${id}.jpg`),
  source: (s: string) => staticFile(`img/${s.replace(/^(gen|input):/, "")}.jpg`),
  sourceThumb: (s: string) => staticFile(`thumb/${s.replace(/^(gen|input):/, "")}.jpg`),
};
export function subscribe() {
  return () => {};
}
