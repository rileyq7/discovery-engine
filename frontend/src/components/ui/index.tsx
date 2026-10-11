/* shadcn/ui-style primitives (copied pattern: cva + tailwind-merge), kept in one file. */
import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { STATIC } from "../../lib/api";
import { cn } from "../../lib/utils";

export const buttonVariants = cva(
  "inline-flex items-center justify-center gap-1.5 whitespace-nowrap rounded-md text-sm font-medium transition-colors disabled:pointer-events-none disabled:opacity-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60 cursor-pointer",
  {
    variants: {
      variant: {
        default: "bg-accent text-accent-fg hover:opacity-90",
        secondary: "bg-panel-2 text-fg border border-border hover:bg-border/60",
        ghost: "text-fg hover:bg-panel-2",
        outline: "border border-border text-fg hover:bg-panel-2",
        destructive: "bg-bad text-white hover:opacity-90",
        good: "bg-good text-black hover:opacity-90",
      },
      size: { default: "h-9 px-3.5", sm: "h-7 px-2.5 text-xs", lg: "h-10 px-5", icon: "h-8 w-8" },
    },
    defaultVariants: { variant: "default", size: "default" },
  },
);

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement>, VariantProps<typeof buttonVariants> {
  /** Still shown in the read-only hosted snapshot (exports, view toggles). Every other button is hidden there. */
  safe?: boolean;
}
export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(({ className, variant, size, safe, ...props }, ref) =>
  STATIC && !safe ? null : <button ref={ref} className={cn(buttonVariants({ variant, size }), className)} {...props} />,
);
Button.displayName = "Button";

export function Card({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("rounded-lg border border-border bg-panel", className)} {...props} />;
}
export function CardHeader({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("flex items-center justify-between gap-2 px-4 pt-3.5 pb-2", className)} {...props} />;
}
export function CardTitle({ className, ...props }: React.HTMLAttributes<HTMLHeadingElement>) {
  return <h3 className={cn("text-sm font-semibold tracking-tight", className)} {...props} />;
}
export function CardContent({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("px-4 pb-4", className)} {...props} />;
}

const badgeVariants = cva("inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium border", {
  variants: {
    tone: {
      neutral: "border-border text-muted bg-panel-2",
      good: "border-good/40 text-good bg-good/10",
      bad: "border-bad/40 text-bad bg-bad/10",
      warn: "border-warn/40 text-warn bg-warn/10",
      info: "border-info/40 text-info bg-info/10",
      accent: "border-accent/40 text-accent bg-accent/10",
    },
  },
  defaultVariants: { tone: "neutral" },
});
export function Badge({ className, tone, ...props }: React.HTMLAttributes<HTMLSpanElement> & VariantProps<typeof badgeVariants>) {
  return <span className={cn(badgeVariants({ tone }), className)} {...props} />;
}

export const Input = React.forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(({ className, ...props }, ref) => (
  <input ref={ref} className={cn("h-9 w-full rounded-md border border-border bg-panel-2 px-2.5 text-sm outline-none focus:border-accent", className)} {...props} />
));
Input.displayName = "Input";

export const Textarea = React.forwardRef<HTMLTextAreaElement, React.TextareaHTMLAttributes<HTMLTextAreaElement>>(({ className, ...props }, ref) => (
  <textarea ref={ref} className={cn("w-full rounded-md border border-border bg-panel-2 p-2.5 text-sm outline-none focus:border-accent", className)} {...props} />
));
Textarea.displayName = "Textarea";

export function Select({ className, ...props }: React.SelectHTMLAttributes<HTMLSelectElement>) {
  return <select className={cn("h-9 rounded-md border border-border bg-panel-2 px-2 text-sm outline-none focus:border-accent", className)} {...props} />;
}

export function Label({ className, ...props }: React.LabelHTMLAttributes<HTMLLabelElement>) {
  return <label className={cn("text-xs font-medium text-muted", className)} {...props} />;
}

export function Table({ className, ...props }: React.TableHTMLAttributes<HTMLTableElement>) {
  return (
    <div className="w-full overflow-x-auto">
      <table className={cn("w-full text-sm", className)} {...props} />
    </div>
  );
}
export function Th({ className, ...props }: React.ThHTMLAttributes<HTMLTableCellElement>) {
  return <th className={cn("h-9 px-3 text-left text-xs font-medium text-muted border-b border-border whitespace-nowrap", className)} {...props} />;
}
export function Td({ className, ...props }: React.TdHTMLAttributes<HTMLTableCellElement>) {
  return <td className={cn("px-3 py-2 border-b border-border/60 align-top", className)} {...props} />;
}

export function Dialog({ open, onClose, title, children, wide }: { open: boolean; onClose: () => void; title: string; children: React.ReactNode; wide?: boolean }) {
  React.useEffect(() => {
    if (!open) return;
    const on = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", on);
    return () => window.removeEventListener("keydown", on);
  }, [open, onClose]);
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-black/60 p-4 pt-[6vh]" onMouseDown={onClose}>
      <div className={cn("w-full rounded-lg border border-border bg-panel shadow-2xl", wide ? "max-w-4xl" : "max-w-lg")} onMouseDown={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between border-b border-border px-4 py-3">
          <h2 className="text-sm font-semibold">{title}</h2>
          <button className="text-muted hover:text-fg cursor-pointer" onClick={onClose} aria-label="Close">✕</button>
        </div>
        <div className="p-4">{children}</div>
      </div>
    </div>
  );
}

export function Empty({ children }: { children: React.ReactNode }) {
  return <div className="rounded-md border border-dashed border-border px-4 py-8 text-center text-sm text-muted">{children}</div>;
}

export function Stat({ label, value, sub }: { label: string; value: React.ReactNode; sub?: React.ReactNode }) {
  return (
    <div>
      <div className="text-xs text-muted">{label}</div>
      <div className="text-xl font-semibold tabular-nums">{value}</div>
      {sub && <div className="text-xs text-muted">{sub}</div>}
    </div>
  );
}

export function Tabs<T extends string>({ value, onChange, options }: { value: T; onChange: (v: T) => void; options: { value: T; label: string }[] }) {
  return (
    <div className="inline-flex rounded-md border border-border bg-panel-2 p-0.5">
      {options.map((o) => (
        <button
          key={o.value}
          onClick={() => onChange(o.value)}
          className={cn("rounded px-2.5 py-1 text-xs font-medium cursor-pointer", value === o.value ? "bg-panel text-fg shadow-sm" : "text-muted hover:text-fg")}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

/** Two-click confirmation instead of a blocking browser dialog. */
export function ConfirmButton({ onConfirm, children, confirmLabel = "Click again to confirm", ...props }: ButtonProps & { onConfirm: () => void; confirmLabel?: string }) {
  const [armed, setArmed] = React.useState(false);
  React.useEffect(() => {
    if (!armed) return;
    const t = setTimeout(() => setArmed(false), 3000);
    return () => clearTimeout(t);
  }, [armed]);
  return (
    <Button {...props} onClick={() => (armed ? (setArmed(false), onConfirm()) : setArmed(true))}>
      {armed ? confirmLabel : children}
    </Button>
  );
}

export function ErrorText({ error }: { error: unknown }) {
  if (!error) return null;
  return <pre className="whitespace-pre-wrap rounded-md border border-bad/40 bg-bad/10 p-2 text-xs text-bad">{String((error as Error).message ?? error)}</pre>;
}
