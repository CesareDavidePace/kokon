import { Images, PanelLeftClose, Pencil, SquarePen, X } from "lucide-react";
import { useState } from "react";
import { urls } from "../lib/api";
import { timeAgo } from "../lib/logic";
import { deleteProject, newProject, renameProject, setState, switchProject, useStore } from "../lib/store";
import { IconButton, NameField } from "./ui";

/** Navigation: start something new, browse every image, or go back to a project. */
export function Sidebar() {
  const projects = useStore((s) => s.projects);
  const current = useStore((s) => s.projectId);
  const empty = useStore((s) => s.session.length === 0);
  const overlay = useStore((s) => s.overlay);
  const gallery = overlay === "gallery";
  const [renaming, setRenaming] = useState<string | null>(null);

  return (
    <aside className="sidebar">
      <div className="side-head">
        <IconButton title="Hide sidebar" onClick={() => setState({ sidebar: false })}>
          <PanelLeftClose size={18} />
        </IconButton>
      </div>
      <nav className="side-nav">
        <button className={`side-item ${empty && !gallery ? "on" : ""}`} onClick={newProject}>
          <SquarePen size={16} /> New image
        </button>
        <button className={`side-item ${gallery ? "on" : ""}`} onClick={() => setState({ overlay: gallery ? null : "gallery" })}>
          <Images size={16} /> All images
        </button>
      </nav>
      {projects.length > 0 && <div className="side-label">Projects</div>}
      <div className="side-list">
        {projects.map((p) => (
          <div
            key={p.id}
            role="button"
            tabIndex={0}
            className={`proj ${p.id === current && !gallery ? "on" : ""}`}
            onClick={() => switchProject(p.id)}
            onDoubleClick={() => setRenaming(p.id)}
            onKeyDown={(e) => {
              if (e.target !== e.currentTarget) return;
              if (e.key === "Enter" || e.key === " ") switchProject(p.id);
              else if (e.key === "F2") setRenaming(p.id);
            }}
            title={`${p.title}
Double-click or F2 to rename`}
          >
            <span className="cover">{p.cover && <img src={urls.sourceThumb(p.cover)} alt="" draggable={false} loading="lazy" />}</span>
            <span className="proj-text">
              {renaming === p.id ? (
                <NameField value={p.named ? p.title : ""} onSave={(v) => renameProject(p.id, v)} onDone={() => setRenaming(null)} />
              ) : (
                <span className="proj-title">{p.title}</span>
              )}
              <span className="proj-sub">
                {p.count} {p.count === 1 ? "version" : "versions"} · {timeAgo(p.updatedAt)}
              </span>
            </span>
            <button
              className="proj-rm"
              title="Rename"
              aria-label="Rename project"
              onClick={(e) => {
                e.stopPropagation();
                setRenaming(p.id);
              }}
            >
              <Pencil size={13} />
            </button>
            <button
              className="proj-rm"
              title="Remove from list (images stay in All images)"
              aria-label="Remove project"
              onClick={(e) => {
                e.stopPropagation();
                deleteProject(p.id);
              }}
            >
              <X size={14} />
            </button>
          </div>
        ))}
      </div>
    </aside>
  );
}

/** The open project's name in the top bar; click to rename (also before the first image). */
export function ProjectTitle() {
  const id = useStore((s) => s.projectId);
  const meta = useStore((s) => s.projects.find((p) => p.id === s.projectId));
  const [editing, setEditing] = useState(false);
  if (editing) return <NameField className="title-field" value={meta?.named ? meta.title : ""} onSave={(v) => renameProject(id, v)} onDone={() => setEditing(false)} />;
  return (
    <button className={`project-title ${meta ? "" : "placeholder"}`} title="Rename project" onClick={() => setEditing(true)}>
      {meta?.title ?? "Untitled project"}
      <Pencil size={13} className="title-pencil" />
    </button>
  );
}
