import { useRef, type ChangeEvent, type DragEvent } from "react";

interface FilePickerProps {
  label: string;
  multiple?: boolean;
  accept?: string;
  files: File[];
  onChange: (files: File[]) => void;
}

export default function FilePicker({ label, multiple, accept = "image/*", files, onChange }: FilePickerProps) {
  const inputRef = useRef<HTMLInputElement>(null);

  function handleFiles(list: FileList | null) {
    if (!list) return;
    onChange(multiple ? Array.from(list) : [list[0]]);
  }

  function onDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    handleFiles(e.dataTransfer.files);
  }

  return (
    <div>
      <p className="mb-1.5 text-sm font-medium text-neutral-700 dark:text-neutral-300">{label}</p>
      <div
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => e.preventDefault()}
        onDrop={onDrop}
        className="cursor-pointer rounded-lg border-2 border-dashed border-neutral-300 bg-neutral-50 px-4 py-6 text-center transition hover:border-neutral-400 dark:border-neutral-700 dark:bg-neutral-900 dark:hover:border-neutral-600"
      >
        <input
          ref={inputRef}
          type="file"
          accept={accept}
          multiple={multiple}
          className="hidden"
          onChange={(e: ChangeEvent<HTMLInputElement>) => handleFiles(e.target.files)}
        />
        {files.length === 0 ? (
          <p className="text-sm text-neutral-500 dark:text-neutral-400">
            Click to choose {multiple ? "images" : "an image"}, or drag and drop
          </p>
        ) : (
          <p className="text-sm text-neutral-700 dark:text-neutral-300">
            {files.map((f) => f.name).join(", ")}
          </p>
        )}
      </div>
    </div>
  );
}
