// Stroke icons (24x24 grid), colored by the text color around them.
import type { ReactNode, SVGProps } from "react";

type IconProps = SVGProps<SVGSVGElement> & { size?: number };

function make(paths: ReactNode) {
  return function Icon({ size = 18, ...rest }: IconProps) {
    return (
      <svg
        width={size}
        height={size}
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth={1.75}
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden="true"
        {...rest}
      >
        {paths}
      </svg>
    );
  };
}

export const PlusIcon = make(<path d="M12 5v14M5 12h14" />);
export const ArrowUpIcon = make(<path d="M12 19V5M5 12l7-7 7 7" />);
export const PanelIcon = make(
  <>
    <rect x="3" y="3" width="18" height="18" rx="2" />
    <path d="M9 3v18" />
  </>,
);
export const SearchIcon = make(
  <>
    <circle cx="11" cy="11" r="7" />
    <path d="m20 20-3.5-3.5" />
  </>,
);
export const InfoIcon = make(
  <>
    <circle cx="12" cy="12" r="9" />
    <path d="M12 16v-4M12 8h.01" />
  </>,
);
export const CopyIcon = make(
  <>
    <rect x="9" y="9" width="12" height="12" rx="2" />
    <path d="M5 15V5a2 2 0 0 1 2-2h10" />
  </>,
);
export const CheckIcon = make(<path d="M20 6 9 17l-5-5" />);
export const XIcon = make(<path d="M18 6 6 18M6 6l12 12" />);
export const RouteIcon = make(
  <>
    <circle cx="5" cy="12" r="2.5" />
    <circle cx="19" cy="5" r="2.5" />
    <circle cx="19" cy="12" r="2.5" />
    <circle cx="19" cy="19" r="2.5" />
    <path d="M7.5 12h9M7.2 10.8l9.6-4.6M7.2 13.2l9.6 4.6" />
  </>,
);
export const SunIcon = make(
  <>
    <circle cx="12" cy="12" r="4" />
    <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
  </>,
);
export const MoonIcon = make(<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />);
export const RetryIcon = make(
  <>
    <path d="M3 12a9 9 0 0 1 15.5-6.2L21 8" />
    <path d="M21 3v5h-5" />
    <path d="M21 12a9 9 0 0 1-15.5 6.2L3 16" />
    <path d="M3 21v-5h5" />
  </>,
);
