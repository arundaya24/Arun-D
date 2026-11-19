import ThemeToggle from "./ThemeToggle";

export default function Navbar() {
  return (
    <header>
      <div>Arun-D</div>
      <nav>
        <a href="#upload">Upload</a>
        <a href="#history">Riwayat</a>
      </nav>
      <ThemeToggle />
    </header>
  );
}