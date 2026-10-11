import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

/** Lab IDs anywhere in the text become links into the app. */
const ID = /\b((?:run|exp|f)-\d{8}-\d{6}-[0-9a-f]{4})\b/g;

export function idHref(id: string): string {
  if (id.startsWith("run-")) return `#/runs/${id}`;
  if (id.startsWith("exp-")) return `#/experiments`;
  return `#/findings`;
}

/** Link bare IDs; leave code spans (linked by the code renderer below) and existing links alone. */
function linkIds(md: string): string {
  return md
    .split(/(```[\s\S]*?```|`[^`\n]*`|\[[^\]\n]*\]\([^)\n]*\))/g)
    .map((part) => {
      if (part.startsWith("`")) return part;
      const link = part.match(/^\[([^\]\n]*)\]\(([^)\n]*)\)$/);
      if (link) {  // an existing link: if it points at a bare lab ID, point it into the app instead
        const target = link[2].trim();
        return /^(run|exp|f)-\d{8}-\d{6}-[0-9a-f]{4}$/.test(target) ? `[${link[1]}](${idHref(target)})` : part;
      }
      return part.replace(ID, (id) => `[${id}](${idHref(id)})`);
    })
    .join("");
}

export function Markdown({ children, className = "prose-de" }: { children: string; className?: string }) {
  return (
    <div className={className}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={{
          code: ({ children: c, className: cls }) => {
            const text = String(c ?? "");
            if (!cls && /^(run|exp|f)-\d{8}-\d{6}-[0-9a-f]{4}$/.test(text)) {
              return <a href={idHref(text)} className="font-mono text-[0.85em] text-accent hover:underline">{text}</a>;
            }
            return <code className={cls}>{c}</code>;
          },
          a: ({ href, children: c }) => (
            <a href={href} className="text-accent hover:underline" {...(href?.startsWith("http") ? { target: "_blank", rel: "noreferrer" } : {})}>{c}</a>
          ),
        }}
      >
        {linkIds(children)}
      </ReactMarkdown>
    </div>
  );
}
