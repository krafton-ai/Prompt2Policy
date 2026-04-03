interface CollapsibleCardProps {
  title: React.ReactNode;
  defaultOpen?: boolean;
  children: React.ReactNode;
}

export default function CollapsibleCard({
  title,
  defaultOpen,
  children,
}: CollapsibleCardProps) {
  return (
    <details
      open={defaultOpen}
      className="bg-white rounded-xl border border-gray-200 shadow-sm"
    >
      <summary className="px-5 py-3 text-sm font-semibold text-gray-900 cursor-pointer hover:bg-gray-50 rounded-xl">
        {title}
      </summary>
      <div className="px-5 pb-5">{children}</div>
    </details>
  );
}
