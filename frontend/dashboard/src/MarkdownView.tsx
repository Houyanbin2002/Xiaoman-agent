import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export function MarkdownView(props: { content: string; className?: string }): React.ReactElement {
  return (
    <div className={`markdown${props.className ? ` ${props.className}` : ""}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        skipHtml
        components={{
          a: ({ href, children, ...linkProps }) => (
            <a href={href} target="_blank" rel="noreferrer" {...linkProps}>
              {children}
            </a>
          ),
        }}
      >
        {props.content}
      </ReactMarkdown>
    </div>
  );
}
