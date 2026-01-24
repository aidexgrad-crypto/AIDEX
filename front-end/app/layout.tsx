import "./globals.css";
import { DatasetProvider } from "./context/DatasetContext";
import ClientShell from "./ClientShell";

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <DatasetProvider>
          <ClientShell>{children}</ClientShell>
        </DatasetProvider>
      </body>
    </html>
  );
}
