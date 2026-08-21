import React, { useEffect, useState } from "react";

export function useAsyncData<T>(loader: () => Promise<T>, dependencies: React.DependencyList): {
  data: T | null;
  loading: boolean;
  error: string;
  reload: () => void;
} {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [revision, setRevision] = useState(0);
  const dependencyKey = JSON.stringify(dependencies);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    setError("");
    void loader()
      .then((value) => {
        if (alive) setData(value);
      })
      .catch((reason: unknown) => {
        if (alive) setError(reason instanceof Error ? reason.message : String(reason));
      })
      .finally(() => {
        if (alive) setLoading(false);
      });
    return () => {
      alive = false;
    };
    // Callers provide the values captured by loader through dependencyKey.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dependencyKey, revision]);

  return { data, loading, error, reload: () => setRevision((value) => value + 1) };
}

