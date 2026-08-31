import fs from "node:fs";

export async function importCfWorkerForNode(workerPath, tag = "test") {
  let src = fs.readFileSync(workerPath, "utf8");
  src = src.replace(
    'import { DurableObject } from "cloudflare:workers";',
    'class DurableObject { constructor(ctx, env) { this.ctx = ctx; this.env = env; } }',
  );
  const encoded = Buffer.from(src, "utf8").toString("base64");
  return import(`data:text/javascript;base64,${encoded}#${encodeURIComponent(tag)}-${Date.now()}`);
}
