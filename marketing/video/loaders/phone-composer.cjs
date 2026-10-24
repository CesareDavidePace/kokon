// The phone's writing screen opens when the prompt is tapped (Composer.tsx keeps that in its own
// state, which a video can't tap): a window's state with `writing: true` opens it too.
module.exports = function (source) {
  const from = "const expanded = !phone || opened || reviewing;";
  if (!source.includes(from)) throw new Error("phone-composer.cjs: Composer.tsx changed, update the loader");
  return source.replace(from, "const expanded = !phone || opened || reviewing || useStore((s) => !!(s as { writing?: boolean }).writing);");
};
