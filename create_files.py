import json, os

base = r"C:\Users\Rodrigo\Documents\Quantitative finance\frontend"
src  = os.path.join(base, "src")
os.makedirs(src, exist_ok=True)

# package.json
with open(os.path.join(base, "package.json"), "w") as f:
    json.dump({
        "name": "ual-nhits",
        "version": "1.0.0",
        "type": "module",
        "scripts": {"dev": "vite", "build": "vite build"},
        "dependencies": {"react": "^18.3.1", "react-dom": "^18.3.1"},
        "devDependencies": {"@vitejs/plugin-react": "^4.3.1", "vite": "^5.4.2"}
    }, f, indent=2)
print("✓ package.json")

# vite.config.js
with open(os.path.join(base, "vite.config.js"), "w") as f:
    f.write(
        'import { defineConfig } from "vite";\n'
        'import react from "@vitejs/plugin-react";\n'
        'export default defineConfig({\n'
        '  plugins: [react()],\n'
        '  server: {\n'
        '    port: 5173,\n'
        '    proxy: { "/api": { target: "http://localhost:5050", changeOrigin: true } }\n'
        '  }\n'
        '});\n'
    )
print("✓ vite.config.js")

# index.html
with open(os.path.join(base, "index.html"), "w") as f:
    f.write(
        '<!DOCTYPE html>\n'
        '<html lang="en">\n'
        '  <head>\n'
        '    <meta charset="UTF-8" />\n'
        '    <meta name="viewport" content="width=device-width, initial-scale=1.0" />\n'
        '    <title>UAL N-HiTS</title>\n'
        '    <style>* { margin: 0; padding: 0; box-sizing: border-box; } body { background: #080810; }</style>\n'
        '  </head>\n'
        '  <body>\n'
        '    <div id="root"></div>\n'
        '    <script type="module" src="/src/main.jsx"></script>\n'
        '  </body>\n'
        '</html>\n'
    )
print("✓ index.html")

# src/main.jsx
with open(os.path.join(src, "main.jsx"), "w") as f:
    f.write(
        'import React from "react";\n'
        'import ReactDOM from "react-dom/client";\n'
        'import App from "./App.jsx";\n'
        'ReactDOM.createRoot(document.getElementById("root")).render(\n'
        '  <React.StrictMode><App /></React.StrictMode>\n'
        ');\n'
    )
print("✓ src/main.jsx")

print("\nTudo criado! Agora corre no terminal:")
print("  cd frontend")
print("  npm install")
print("  npm run dev")