import * as React from "react";
import * as ReactJSXRuntime from "react/jsx-runtime";
import * as ReactDOMClient from "react-dom/client";
import * as UI from "./sdk";

// The shared runtime handed to dynamically-imported plugin modules. The static
// shim files under /assets/sdk/*.js read these off window so that plugins and
// the host resolve react / react-dom / @xiaoman/dashboard-ui to one instance.
export interface XiaomanRuntime {
  React: typeof React;
  ReactJSXRuntime: typeof ReactJSXRuntime;
  ReactDOMClient: typeof ReactDOMClient;
  UI: typeof UI;
}

declare global {
  interface Window {
    __xiaomanRuntime?: XiaomanRuntime;
    /** Compatibility alias for already-published dashboard plugins. */
    __akashicRuntime?: XiaomanRuntime;
  }
}

// Publish the runtime before any plugin is imported.
export function exposeRuntime(): void {
  window.__xiaomanRuntime = { React, ReactJSXRuntime, ReactDOMClient, UI };
  window.__akashicRuntime = window.__xiaomanRuntime;
}
