// Runs a page in headless Chrome (WebGPU enabled) against a temporary Vite dev server.
// Usage: node scripts/e2e.mjs [path=/test.html] [--headful] [--shot out.png] [--eval js]
// For /index.html (the Ladder page), set HF_TOKEN to seed the page with a token for the private ladder repos.
import { createServer } from "vite";
import puppeteer from "puppeteer-core";

const args = process.argv.slice(2);
const path = args.find((a) => a.startsWith("/")) ?? "/test.html";
const headful = args.includes("--headful");
const shotIdx = args.indexOf("--shot");
const shot = shotIdx >= 0 ? args[shotIdx + 1] : null;
const evalIdx = args.indexOf("--eval");
const evalJs = evalIdx >= 0 ? args[evalIdx + 1] : null;

const server = await createServer({ server: { port: 5199, strictPort: false }, logLevel: "warn" });
await server.listen();
const url = server.resolvedUrls.local[0].replace(/\/$/, "") + path;

const browser = await puppeteer.launch({
  executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  headless: !headful,
  args: ["--enable-unsafe-webgpu", "--enable-features=WebGPU", "--window-size=1600,1000"],
  defaultViewport: { width: 1600, height: 1000, deviceScaleFactor: 1 },
});
let code = 0;
try {
  const page = await browser.newPage();
  page.on("console", (m) => console.log(`[page] ${m.text()}`));
  page.on("pageerror", (e) => console.log(`[pageerror] ${e.message}`));
  if (process.env.HF_TOKEN) {
    await page.evaluateOnNewDocument((t) => localStorage.setItem("nanogpt-ladder:hf-token", t), process.env.HF_TOKEN);
  }
  await page.goto(url);
  if (path.startsWith("/test.html")) {
    await page.waitForFunction("window.__selftest", { timeout: 600_000 });
    const r = await page.evaluate("window.__selftest");
    console.log(JSON.stringify(r));
    code = r.pass ? 0 : 1;
  } else {
    await page.waitForFunction("window.__ready === true", { timeout: 600_000 });
    if (evalJs) console.log("eval:", JSON.stringify(await page.evaluate(evalJs)));
    await page.evaluate("new Promise(r => setTimeout(r, 500))");
  }
  if (shot) await page.screenshot({ path: shot, fullPage: false });
} catch (e) {
  console.error(e);
  code = 1;
} finally {
  await browser.close();
  await server.close();
}
process.exit(code);
