import type { DLTable } from "../../lib/deepLearningProjects";

export default function DataTable({ table }: { table: DLTable }) {
  return (
    <figure>
      <div className="overflow-x-auto rounded-lg border border-neutral-200 dark:border-neutral-800">
        <table className="w-full min-w-[420px] border-collapse text-sm">
          <thead>
            <tr className="border-b border-neutral-200 bg-neutral-50 dark:border-neutral-800 dark:bg-neutral-900">
              {table.headers.map((h) => (
                <th key={h} className="px-3 py-2 text-left font-medium text-neutral-600 dark:text-neutral-400">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {table.rows.map((row, i) => (
              <tr key={i} className="border-b border-neutral-100 last:border-0 dark:border-neutral-800/60">
                {row.map((cell, j) => (
                  <td
                    key={j}
                    className={`px-3 py-2 ${j === 0 ? "font-medium text-neutral-800 dark:text-neutral-200" : "text-neutral-600 dark:text-neutral-400"}`}
                  >
                    {cell}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {table.caption && <figcaption className="mt-1.5 text-xs text-neutral-500">{table.caption}</figcaption>}
    </figure>
  );
}
