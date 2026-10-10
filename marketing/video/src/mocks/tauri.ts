// Stand-ins for the Tauri APIs: nothing native runs inside a video.
const never = () => new Promise<never>(() => {});
export const invoke = never;
export const getCurrentWindow = () => ({ isFullscreen: never, setFullscreen: never });
export const getCurrentWebview = () => ({ onDragDropEvent: never });
export const ask = never;
export const open = never;
export const save = never;
export const writeText = never;
export const relaunch = never;
export const check = never;
export const openUrl = never;
export type Update = unknown;
