import { Dices, RotateCcw } from "lucide-react";
import { planPipeline } from "../lib/logic";
import { updateComposer, useStore } from "../lib/store";
import type { Advanced, OutputFormat, Plan, ResolutionMode } from "../lib/types";
import { Seg, Switch } from "./ui";

// Verified against sd-server 3f8527a capabilities; replaced by the live list when loaded.
const FALLBACK_SAMPLERS = ["euler", "euler_a", "heun", "dpm2", "dpm++2s_a", "dpm++2m", "dpm++2mv2", "ipndm", "ipndm_v", "lcm", "ddim_trailing", "tcd", "res_multistep", "res_2s", "er_sde"];
const FALLBACK_SCHEDULERS = ["discrete", "karras", "exponential", "ays", "gits", "sgm_uniform", "simple", "smoothstep", "kl_optimal", "lcm", "beta", "flux"];

const RESOLUTIONS: { value: ResolutionMode; label: string }[] = [
  { value: "auto", label: "Auto" },
  { value: "source", label: "Source" },
  { value: "1k", label: "1K" },
  { value: "1.5k", label: "1.5K" },
  { value: "2k", label: "2K" },
  { value: "custom", label: "Custom" },
];

function NumField(props: { label: string; value: number | null | undefined; placeholder: string; min: number; max: number; step?: number; onChange: (v: number | null) => void }) {
  return (
    <label className="field-row">
      <span>{props.label}</span>
      <input
        className="input"
        style={{ width: 110 }}
        type="number"
        min={props.min}
        max={props.max}
        step={props.step ?? 1}
        placeholder={props.placeholder}
        value={props.value ?? ""}
        onChange={(e) => {
          const v = e.target.value === "" ? null : Number(e.target.value);
          props.onChange(v === null || Number.isNaN(v) ? null : Math.min(props.max, Math.max(props.min, v)));
        }}
      />
    </label>
  );
}

export function AdvancedPanel({ plan }: { plan: Plan | null }) {
  const c = useStore((s) => s.composer);
  const samplers = useStore((s) => (s.status?.server as { samplers?: string[] } | undefined)?.samplers);
  const schedulers = useStore((s) => (s.status?.server as { schedulers?: string[] } | undefined)?.schedulers);
  const a = c.advanced;
  const setA = (patch: Partial<Advanced>) => updateComposer({ advanced: { ...a, ...patch } });
  const editLike = plan && (plan.mode === "inpaint" || plan.mode === "outpaint" || plan.mode === "edit");

  return (
    <>
      <div className="field">
        <span>Resolution</span>
        <Seg value={c.resolution} options={RESOLUTIONS} onChange={(v) => updateComposer({ resolution: v })} />
        {c.resolution === "custom" && (
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <input className="input" type="number" min={64} max={4096} value={c.customWidth} onChange={(e) => updateComposer({ customWidth: Math.max(64, Math.min(4096, Number(e.target.value) || 64)) })} style={{ width: 90 }} />
            <span className="muted">×</span>
            <input className="input" type="number" min={64} max={4096} value={c.customHeight} onChange={(e) => updateComposer({ customHeight: Math.max(64, Math.min(4096, Number(e.target.value) || 64)) })} style={{ width: 90 }} />
          </div>
        )}
        {plan && <span className="muted">{planPipeline(plan)}</span>}
      </div>

      <label className="field-row">
        <span>Seed</span>
        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          <input
            className="input"
            style={{ width: 120 }}
            type="number"
            placeholder="Random"
            value={c.seed ?? ""}
            onChange={(e) => updateComposer({ seed: e.target.value === "" ? null : Math.max(0, Math.floor(Number(e.target.value))) })}
          />
          <button className="icon-btn sm" title="Random seed" onClick={() => updateComposer({ seed: null })}>
            <Dices size={15} />
          </button>
        </div>
      </label>

      <NumField label="Steps" value={a.steps} placeholder={String(plan?.steps ?? "Auto")} min={1} max={150} onChange={(v) => setA({ steps: v })} />
      <NumField label="CFG" value={a.cfg} placeholder={String(plan?.cfg ?? "Auto")} min={0} max={30} step={0.5} onChange={(v) => setA({ cfg: v })} />

      <label className="field-row">
        <span>Sampler</span>
        <select className="select" value={a.sampler ?? ""} onChange={(e) => setA({ sampler: e.target.value || null })}>
          <option value="">Default ({plan?.sampler ?? "euler"})</option>
          {(samplers?.length ? samplers : FALLBACK_SAMPLERS).map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
      </label>
      <label className="field-row">
        <span>Scheduler</span>
        <select className="select" value={a.scheduler ?? ""} onChange={(e) => setA({ scheduler: e.target.value || null })}>
          <option value="">Default (model)</option>
          {(schedulers?.length ? schedulers : FALLBACK_SCHEDULERS).map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
      </label>

      {editLike && (
        <label className="field">
          <span>
            Edit strength {a.strength != null ? a.strength.toFixed(2) : plan?.strength != null ? `(auto ${plan.strength})` : "(auto)"}
          </span>
          <input type="range" min={0.05} max={1} step={0.05} value={a.strength ?? plan?.strength ?? 1} onChange={(e) => setA({ strength: Number(e.target.value) })} />
        </label>
      )}

      <NumField label="Images per run" value={a.batch_count} placeholder="1" min={1} max={4} onChange={(v) => setA({ batch_count: v })} />

      <div className="field">
        <span>Format</span>
        <Seg<OutputFormat>
          value={a.output_format ?? "png"}
          options={[
            { value: "png", label: "PNG" },
            { value: "jpeg", label: "JPEG" },
            { value: "webp", label: "WebP" },
          ]}
          onChange={(v) => setA({ output_format: v })}
        />
      </div>

      {c.refs[0]?.hdr && (
        <div className="field-row">
          <span title="The HEIC source has an HDR gain map. The result is saved as an Ultra HDR JPEG; the HDR boost fades where the edit changed brightness.">
            Keep HDR (Ultra HDR JPEG)
          </span>
          <Switch on={a.hdr ?? true} onChange={(v) => setA({ hdr: v })} label="Keep HDR" />
        </div>
      )}

      <div className="field-row">
        <span>
          VAE tiling {a.vae_tiling == null ? `(auto: ${plan?.vae_tiling ? "on" : "off"})` : ""}
        </span>
        <Switch on={a.vae_tiling ?? plan?.vae_tiling ?? false} onChange={(v) => setA({ vae_tiling: v })} label="VAE tiling" />
      </div>

      <label className="field">
        <span>Negative prompt</span>
        <textarea className="input" rows={2} value={c.negative} placeholder="Optional; used only when CFG is above 1" onChange={(e) => updateComposer({ negative: e.target.value })} />
      </label>

      <button className="btn" onClick={() => updateComposer({ advanced: {}, seed: null, negative: "", resolution: "auto" })}>
        <RotateCcw size={14} /> Reset to automatic
      </button>
    </>
  );
}
