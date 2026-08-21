import React, { useEffect, useMemo, useRef, useState } from "react";
import { Brain, Maximize2, Minus, Plus, X } from "lucide-react";
import { relativeTime } from "../../format";
import type {
  MemoryKnowledgeEdge,
  MemoryKnowledgeGraph as MemoryKnowledgeGraphData,
  MemoryKnowledgeNode,
  PersonalRecordRow,
} from "../../shared/types";

interface Point {
  x: number;
  y: number;
}

type Interaction =
  | { mode: "pan"; pointerId: number; clientX: number; clientY: number }
  | { mode: "node"; pointerId: number; nodeId: string; clientX: number; clientY: number }
  | null;

const WIDTH = 1120;
const HEIGHT = 620;
const MIN_SCALE = 0.58;
const MAX_SCALE = 2.2;

const KIND_LABELS: Record<string, string> = {
  self: "关于我",
  requested: "明确记住",
  fact: "事实",
  preference: "偏好",
  temporary_state: "当前状态",
  historical_event: "重要经历",
  episode: "重要经历",
  relationship: "关系",
  procedure: "助手上下文",
  entity: "关联对象",
};

const KIND_ANGLES: Record<string, number> = {
  relationship: -2.55,
  preference: -1.55,
  fact: -0.6,
  requested: 0.15,
  temporary_state: 0.85,
  historical_event: 1.55,
  episode: 1.55,
  procedure: 2.35,
  entity: 2.85,
};

export function MemoryKnowledgeGraph(props: {
  graph: MemoryKnowledgeGraphData | null;
  records: PersonalRecordRow[];
  loading: boolean;
  query: string;
}): React.ReactElement {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const interaction = useRef<Interaction>(null);
  const [positions, setPositions] = useState<Record<string, Point>>({});
  const [view, setView] = useState({ x: 0, y: 0, scale: 1 });
  const [selectedId, setSelectedId] = useState("");
  const graph = props.graph;

  useEffect(() => {
    if (!graph) return;
    const layout = createLayout(graph.nodes, graph.center_id);
    setPositions((current) => Object.fromEntries(
      graph.nodes.map((node) => [node.id, current[node.id] ?? layout[node.id]]),
    ));
    if (selectedId && !graph.nodes.some((node) => node.id === selectedId)) {
      setSelectedId("");
    }
  }, [graph, selectedId]);

  const recordById = useMemo(
    () => new Map(props.records.map((record) => [record.id, record])),
    [props.records],
  );
  const selected = graph?.nodes.find((node) => node.id === selectedId) ?? null;
  const selectedRecords = selected?.memory_ids
    .map((id) => recordById.get(id))
    .filter((record): record is PersonalRecordRow => Boolean(record)) ?? [];
  const normalizedQuery = props.query.trim().toLocaleLowerCase();
  const matches = useMemo(() => {
    const result = new Set<string>();
    if (!graph || !normalizedQuery) return result;
    for (const node of graph.nodes) {
      const relatedText = node.memory_ids
        .map((id) => recordById.get(id))
        .map((record) => record ? `${record.title} ${record.summary} ${String(record.data.content ?? "")}` : "")
        .join(" ");
      if (`${node.label} ${relatedText}`.toLocaleLowerCase().includes(normalizedQuery)) {
        result.add(node.id);
      }
    }
    return result;
  }, [graph, normalizedQuery, recordById]);

  const resetView = (): void => {
    if (graph) setPositions(createLayout(graph.nodes, graph.center_id));
    setView({ x: 0, y: 0, scale: 1 });
  };

  const zoom = (delta: number): void => {
    setView((current) => ({
      ...current,
      scale: clamp(current.scale + delta, MIN_SCALE, MAX_SCALE),
    }));
  };

  const coordinateDelta = (clientX: number, clientY: number, previousX: number, previousY: number): Point => {
    const rect = svgRef.current?.getBoundingClientRect();
    return {
      x: (clientX - previousX) * (rect ? WIDTH / rect.width : 1),
      y: (clientY - previousY) * (rect ? HEIGHT / rect.height : 1),
    };
  };

  const onPointerMove = (event: React.PointerEvent<SVGSVGElement>): void => {
    const active = interaction.current;
    if (!active || active.pointerId !== event.pointerId) return;
    const delta = coordinateDelta(event.clientX, event.clientY, active.clientX, active.clientY);
    if (active.mode === "node") {
      setPositions((current) => ({
        ...current,
        [active.nodeId]: {
          x: (current[active.nodeId]?.x ?? WIDTH / 2) + delta.x / view.scale,
          y: (current[active.nodeId]?.y ?? HEIGHT / 2) + delta.y / view.scale,
        },
      }));
    } else {
      setView((current) => ({ ...current, x: current.x + delta.x, y: current.y + delta.y }));
    }
    interaction.current = { ...active, clientX: event.clientX, clientY: event.clientY };
  };

  const stopInteraction = (event: React.PointerEvent<SVGSVGElement>): void => {
    if (interaction.current?.pointerId === event.pointerId) interaction.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  };

  if (props.loading && !graph) {
    return <div className="memory-graph-loading"><Brain size={25} /><span>正在整理记忆关系…</span></div>;
  }
  if (!graph || graph.nodes.length <= 1) {
    return <div className="memory-graph-empty"><span><Brain size={28} /></span><h2>还没有形成记忆关系</h2><p>继续告诉小满你的偏好、重要关系和经历，关联会自然出现在这里。</p></div>;
  }

  return <section className="memory-knowledge-graph" aria-label="个人记忆知识图谱">
    <div className="memory-graph-heading">
      <div><small>你的长期记忆网络</small><h2>小满怎样理解你</h2><p>拖动节点整理位置，拖动空白处浏览整张图。</p></div>
      <div className="memory-graph-controls" aria-label="图谱视图控制">
        <button type="button" onClick={() => zoom(-0.14)} aria-label="缩小图谱"><Minus size={15} /></button>
        <span>{Math.round(view.scale * 100)}%</span>
        <button type="button" onClick={() => zoom(0.14)} aria-label="放大图谱"><Plus size={15} /></button>
        <button type="button" onClick={resetView} aria-label="恢复图谱视图"><Maximize2 size={15} /></button>
      </div>
    </div>

    <div className="memory-graph-stage">
      <svg
        ref={svgRef}
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label={`包含 ${graph.nodes.length} 个节点和 ${graph.edges.length} 条关系的记忆图谱`}
        onPointerDown={(event) => {
          if (event.target !== event.currentTarget) return;
          interaction.current = { mode: "pan", pointerId: event.pointerId, clientX: event.clientX, clientY: event.clientY };
          event.currentTarget.setPointerCapture(event.pointerId);
        }}
        onPointerMove={onPointerMove}
        onPointerUp={stopInteraction}
        onPointerCancel={stopInteraction}
        onDoubleClick={resetView}
        onWheel={(event) => {
          event.preventDefault();
          zoom(event.deltaY > 0 ? -0.1 : 0.1);
        }}
      >
        <defs>
          <filter id="memory-node-glow" x="-70%" y="-70%" width="240%" height="240%">
            <feGaussianBlur stdDeviation="7" result="blur" />
            <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
        </defs>
        <g transform={`translate(${view.x} ${view.y}) scale(${view.scale})`}>
          {graph.edges.map((edge) => <GraphEdgeView key={edge.id} edge={edge} positions={positions} selectedId={selectedId} dimmed={Boolean(normalizedQuery) && !matches.has(edge.source) && !matches.has(edge.target)} />)}
          {graph.nodes.map((node) => {
            const point = positions[node.id] ?? { x: WIDTH / 2, y: HEIGHT / 2 };
            const selectedNode = node.id === selectedId;
            const dimmed = Boolean(normalizedQuery) && !matches.has(node.id);
            const radius = node.id === graph.center_id ? 38 : Math.min(28, 20 + node.memory_ids.length * 1.5);
            return <g
              key={node.id}
              className={`memory-graph-node kind-${kindClass(node.kind)}${selectedNode ? " selected" : ""}${dimmed ? " dimmed" : ""}`}
              transform={`translate(${point.x} ${point.y})`}
              role="button"
              tabIndex={0}
              aria-label={`${node.label}，${KIND_LABELS[node.kind] ?? "记忆节点"}`}
              onKeyDown={(event) => {
                if (event.key !== "Enter" && event.key !== " ") return;
                event.preventDefault();
                setSelectedId(node.id);
              }}
              onPointerDown={(event) => {
                event.stopPropagation();
                setSelectedId(node.id);
                interaction.current = { mode: "node", pointerId: event.pointerId, nodeId: node.id, clientX: event.clientX, clientY: event.clientY };
                event.currentTarget.ownerSVGElement?.setPointerCapture(event.pointerId);
              }}
            >
              <circle className="memory-graph-node-halo" r={radius + 9} />
              <circle className="memory-graph-node-core" r={radius} filter={selectedNode ? "url(#memory-node-glow)" : undefined} />
              {node.id === graph.center_id ? <Brain className="memory-graph-node-icon" x={-13} y={-13} width={26} height={26} /> : null}
              <text className="memory-graph-node-label" y={radius + 23} textAnchor="middle">{truncate(node.label, 12)}</text>
              {node.memory_ids.length > 1 ? <text className="memory-graph-node-count" x={radius - 2} y={-radius + 5}>{node.memory_ids.length}</text> : null}
              <title>{node.label}</title>
            </g>;
          })}
        </g>
      </svg>

      {selected ? <aside className="memory-graph-detail" aria-live="polite">
        <button type="button" className="memory-graph-detail-close" onClick={() => setSelectedId("")} aria-label="关闭节点详情"><X size={15} /></button>
        <small>{KIND_LABELS[selected.kind] ?? "关联记忆"}</small>
        <h3>{selected.label}</h3>
        {selected.id === graph.center_id ? <><p>这是整张图的中心。周围节点来自小满当前有效的长期记忆。</p><div className="memory-graph-detail-meta"><span>{graph.nodes.length - 1} 个关联</span><span>{graph.edges.length} 条关系</span></div></> : selectedRecords.length ? selectedRecords.slice(0, 3).map((record) => <div className="memory-graph-record" key={record.id}><strong>{record.title}</strong>{String(record.data.content ?? "") && String(record.data.content) !== record.title ? <p>{String(record.data.content)}</p> : null}<div className="memory-graph-detail-meta"><span>{Math.round(record.confidence * 100)}% 可信</span><span>更新于 {relativeTime(record.updated_at)}</span>{record.expires_at ? <span>有效至 {relativeTime(record.expires_at)}</span> : <span>长期有效</span>}</div></div>) : <p>这条关系来自当前长期记忆。</p>}
      </aside> : null}
    </div>
  </section>;
}

function GraphEdgeView(props: {
  edge: MemoryKnowledgeEdge;
  positions: Record<string, Point>;
  selectedId: string;
  dimmed: boolean;
}): React.ReactElement | null {
  const source = props.positions[props.edge.source];
  const target = props.positions[props.edge.target];
  if (!source || !target) return null;
  const highlighted = props.edge.source === props.selectedId || props.edge.target === props.selectedId;
  const middleX = (source.x + target.x) / 2;
  const middleY = (source.y + target.y) / 2;
  return <g className={`memory-graph-edge${highlighted ? " highlighted" : ""}${props.dimmed ? " dimmed" : ""}`}>
    <line x1={source.x} y1={source.y} x2={target.x} y2={target.y} />
    {highlighted ? <text x={middleX} y={middleY - 7} textAnchor="middle">{truncate(props.edge.label, 10)}</text> : null}
  </g>;
}

function createLayout(nodes: MemoryKnowledgeNode[], centerId: string): Record<string, Point> {
  const result: Record<string, Point> = { [centerId]: { x: WIDTH / 2, y: HEIGHT / 2 } };
  const peers = nodes
    .filter((node) => node.id !== centerId)
    .sort((left, right) => {
      const kindOrder = (KIND_ANGLES[left.kind] ?? 10) - (KIND_ANGLES[right.kind] ?? 10);
      return kindOrder || left.label.localeCompare(right.label, "zh-CN");
    });
  const perRing = 14;
  for (let offset = 0, ring = 0; offset < peers.length; offset += perRing, ring += 1) {
    const ringNodes = peers.slice(offset, offset + perRing);
    const radiusX = 252 + ring * 112;
    const radiusY = 202 + ring * 78;
    ringNodes.forEach((node, index) => {
      const angle = -Math.PI / 2 + ring * 0.13 + index * Math.PI * 2 / ringNodes.length;
      result[node.id] = {
        x: WIDTH / 2 + Math.cos(angle) * radiusX,
        y: HEIGHT / 2 + Math.sin(angle) * radiusY,
      };
    });
  }
  return result;
}

function kindClass(kind: string): string {
  return kind.replace(/[^a-z0-9_-]/gi, "-").toLocaleLowerCase();
}

function truncate(value: string, limit: number): string {
  return value.length > limit ? `${value.slice(0, limit)}…` : value;
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}
