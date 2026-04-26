import type { Candidate } from "./types";

type Props = {
  candidate: Candidate;
};

export default function ScoreBreakdown({ candidate }: Props) {
  const s = candidate.score_breakdown;

  if (!s) return null;

  const total =
    s.relevance +
    s.redundancy +
    s.coverage +
    s.length_penalty +
    s.metadata_bonus;

  const segments = [
    { label: "rel", value: s.relevance, color: "#4ade80" },
    { label: "red", value: s.redundancy, color: "#ef4444" },
    { label: "cov", value: s.coverage, color: "#60a5fa" },
    { label: "len", value: s.length_penalty, color: "#f59e0b" },
    { label: "meta", value: s.metadata_bonus, color: "#a855f7" },
  ];

  return (
    <div style={{ marginTop: 10 }}>
      <div style={{ fontSize: 12, marginBottom: 4 }}>
        Score: {total.toFixed(3)}
      </div>

      {/* BAR */}
      <div
        style={{
          display: "flex",
          height: 10,
          width: "100%",
          borderRadius: 4,
          overflow: "hidden",
          background: "#eee",
        }}
      >
        {segments.map((seg, i) => {
          const width = Math.max(Math.abs(seg.value) * 100, 0);

          return (
            <div
              key={i}
              title={`${seg.label}: ${seg.value.toFixed(3)}`}
              style={{
                width: `${width}%`,
                backgroundColor: seg.color,
                opacity: seg.value < 0 ? 0.5 : 1,
              }}
            />
          );
        })}
      </div>

      {/* LABELS */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          fontSize: 10,
          marginTop: 4,
        }}
      >
        {segments.map((s) => (
          <span key={s.label}>
            {s.label}: {s.value.toFixed(2)}
          </span>
        ))}
      </div>
    </div>
  );
}
