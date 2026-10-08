import { ArrowLeft, Heart, Loader2, Search, Trash2 } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { api, urls } from "../lib/api";
import { native } from "../lib/native";
import { forgetGeneration, openGeneration, setState, toastError, useStore } from "../lib/store";
import type { GenerationRecord } from "../lib/types";
import { IconButton, Seg } from "./ui";

const PAGE = 60;

export function Gallery() {
  const version = useStore((s) => s.galleryVersion);
  const [q, setQ] = useState("");
  const [favorites, setFavorites] = useState(false);
  const [sort, setSort] = useState<"recent" | "oldest">("recent");
  const [items, setItems] = useState<GenerationRecord[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const gridRef = useRef<HTMLDivElement>(null);

  const load = useCallback(
    async (offset: number) => {
      setLoading(true);
      try {
        const r = await api.gallery({ q, favorites, sort, limit: PAGE, offset });
        setItems((prev) => (offset === 0 ? r.items : [...prev, ...r.items]));
        setTotal(r.total);
      } catch (e) {
        toastError(e);
      } finally {
        setLoading(false);
      }
    },
    [q, favorites, sort],
  );

  useEffect(() => {
    const t = window.setTimeout(() => load(0), q ? 250 : 0);
    return () => window.clearTimeout(t);
  }, [load, version, q]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setState({ overlay: null });
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const reopen = (g: GenerationRecord) => openGeneration(g).catch(toastError);

  const toggleFav = async (g: GenerationRecord) => {
    try {
      const r = await api.setFavorite(g.id, !g.favorite);
      setItems((list) => (favorites && !r.favorite ? list.filter((x) => x.id !== g.id) : list.map((x) => (x.id === g.id ? r : x))));
    } catch (e) {
      toastError(e);
    }
  };

  const remove = async (g: GenerationRecord) => {
    try {
      await api.deleteGeneration(g.id);
      setItems((list) => list.filter((x) => x.id !== g.id));
      setTotal((t) => t - 1);
      forgetGeneration(g.id);
    } catch (e) {
      toastError(e);
    }
  };

  return (
    <div className="overlay">
      <div className="overlay-head">
        <IconButton title="Back (Esc)" onClick={() => setState({ overlay: null })}>
          <ArrowLeft size={18} />
        </IconButton>
        <h2>Gallery</h2>
        <span className="muted">{total}</span>
      </div>
      <div className="gallery-bar">
        <label className="search">
          <Search size={15} />
          <input type="text" placeholder="Search prompts" value={q} onChange={(e) => setQ(e.target.value)} autoFocus />
        </label>
        <IconButton title={favorites ? "Show all" : "Favorites only"} active={favorites} onClick={() => setFavorites((f) => !f)}>
          <Heart size={17} fill={favorites ? "currentColor" : "none"} />
        </IconButton>
        <div style={{ width: 170 }}>
          <Seg
            value={sort}
            options={[
              { value: "recent", label: "Recent" },
              { value: "oldest", label: "Oldest" },
            ]}
            onChange={setSort}
          />
        </div>
      </div>
      <div
        className="grid"
        ref={gridRef}
        onScroll={(e) => {
          const el = e.currentTarget;
          if (!loading && items.length < total && el.scrollTop + el.clientHeight > el.scrollHeight - 400) load(items.length);
        }}
      >
        {items.map((g) => (
          <div key={g.id} className="card" onClick={() => reopen(g)} title={g.prompt}>
            <img src={urls.genThumb(g.id)} alt="" loading="lazy" draggable={false} />
            <div className={`card-actions ${g.favorite ? "fav" : ""}`} onClick={(e) => e.stopPropagation()}>
              <button title={g.favorite ? "Remove from favorites" : "Favorite"} onClick={() => toggleFav(g)}>
                <Heart size={14} fill={g.favorite ? "currentColor" : "none"} />
              </button>
              <button
                title="Delete"
                onClick={async () => {
                  if (await native.confirm("Delete this image from the gallery and disk?", "Delete")) remove(g);
                }}
              >
                <Trash2 size={14} />
              </button>
            </div>
            <div className="card-meta">{g.prompt}</div>
          </div>
        ))}
        {!loading && items.length === 0 && <div className="muted" style={{ gridColumn: "1/-1", textAlign: "center", padding: 48 }}>{q || favorites ? "Nothing matches." : "Your generations will appear here."}</div>}
        {loading && (
          <div style={{ gridColumn: "1/-1", display: "grid", placeItems: "center", padding: 24 }}>
            <Loader2 className="spin" size={20} />
          </div>
        )}
      </div>
    </div>
  );
}
