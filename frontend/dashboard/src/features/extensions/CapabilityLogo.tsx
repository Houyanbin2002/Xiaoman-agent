import React, { useEffect, useState } from "react";
import { Cable } from "lucide-react";
import type { CapabilityBrand } from "./capabilityPresentation";

interface CapabilityLogoProps {
  brand: CapabilityBrand;
  size?: "sm" | "md" | "lg";
}

export function CapabilityLogo({
  brand,
  size = "md",
}: CapabilityLogoProps): React.ReactElement {
  const [failed, setFailed] = useState(false);

  useEffect(() => setFailed(false), [brand.logo]);

  return (
    <span
      className={`capability-logo size-${size}`}
      style={{ "--logo-accent": brand.accent } as React.CSSProperties}
      aria-hidden="true"
    >
      {brand.logo && !failed
        ? <img src={brand.logo} alt="" onError={() => setFailed(true)} />
        : <Cable />}
    </span>
  );
}
