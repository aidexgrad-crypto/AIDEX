import "./globals.css";
import { DatasetProvider } from "./context/DatasetContext";
import ClientShell from "./ClientShell";

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-[#0b1020] text-white min-h-screen">
        <DatasetProvider>
          <ClientShell>{children}</ClientShell>
        </DatasetProvider>
      </body>
    </html>
  );
}
