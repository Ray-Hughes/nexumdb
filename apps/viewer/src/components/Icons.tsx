/**
 * Icons, drawn inline.
 *
 * A handful of 16px glyphs is not worth an icon-font dependency, and inline
 * SVG inherits `currentColor` so a nav item's active state colours its icon
 * without any extra wiring.
 */

interface IconProps {
  className?: string;
}

const base = {
  viewBox: "0 0 16 16",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.5,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
};

export function OverviewIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <rect x="2" y="2" width="5" height="5" rx="1" />
      <rect x="9" y="2" width="5" height="5" rx="1" />
      <rect x="2" y="9" width="5" height="5" rx="1" />
      <rect x="9" y="9" width="5" height="5" rx="1" />
    </svg>
  );
}

export function CollectionIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M3 2.5h7l3 3v8a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1v-10a1 1 0 0 1 1-1Z" />
      <path d="M9.5 2.5v3.5h3.5" />
    </svg>
  );
}

export function ChunkIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <rect x="2" y="3" width="12" height="3" rx="1" />
      <rect x="2" y="10" width="12" height="3" rx="1" />
      <path d="M5 6.5v3" />
    </svg>
  );
}

export function SearchIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <circle cx="7" cy="7" r="4.5" />
      <path d="m10.5 10.5 3 3" />
    </svg>
  );
}

export function GraphIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <circle cx="3.5" cy="4" r="1.8" />
      <circle cx="12.5" cy="6" r="1.8" />
      <circle cx="7" cy="12.5" r="1.8" />
      <path d="M5.2 4.8 10.8 5.6M11.7 7.6 8.2 10.9M6.2 10.8 4.2 5.8" />
    </svg>
  );
}

export function ScatterIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M2 2v12h12" />
      <circle cx="5.5" cy="10.5" r="1.1" fill="currentColor" stroke="none" />
      <circle cx="8" cy="7" r="1.1" fill="currentColor" stroke="none" />
      <circle cx="11" cy="9" r="1.1" fill="currentColor" stroke="none" />
      <circle cx="12.5" cy="4.5" r="1.1" fill="currentColor" stroke="none" />
    </svg>
  );
}

export function HistoryIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <circle cx="8" cy="8" r="6" />
      <path d="M8 4.5V8l2.5 1.5" />
    </svg>
  );
}

export function FolderIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M2 4a1 1 0 0 1 1-1h3l1.5 2H13a1 1 0 0 1 1 1v6a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V4Z" />
    </svg>
  );
}

export function CloseIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="m4 4 8 8M12 4l-8 8" />
    </svg>
  );
}

export function ExpandIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M6 3H3v3M10 13h3v-3M13 6V3h-3M3 10v3h3" />
    </svg>
  );
}

export function WarningIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M8 2.5 14.5 13.5h-13L8 2.5Z" />
      <path d="M8 6.5v3.2M8 11.6v.01" />
    </svg>
  );
}

/** The product mark: three linked nodes — the data model in miniature. */
export function BrandMark(props: IconProps) {
  return (
    <svg viewBox="0 0 24 24" fill="none" {...props}>
      <path
        d="M6.5 7.5 17 5.5M17.5 8.5 8.5 17M6.5 9.5 7 15"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinecap="round"
        opacity="0.55"
      />
      <circle cx="5.5" cy="6.5" r="2.6" fill="currentColor" />
      <circle cx="18.5" cy="5" r="2" fill="currentColor" opacity="0.8" />
      <circle cx="7.5" cy="18" r="2.2" fill="currentColor" opacity="0.65" />
      <circle cx="18" cy="9.5" r="1.7" fill="currentColor" opacity="0.5" />
    </svg>
  );
}
