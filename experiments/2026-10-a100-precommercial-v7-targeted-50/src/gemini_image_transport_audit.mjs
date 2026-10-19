// Audit only model-image requests. Never log headers, tokens, prompt bodies or OAuth exchanges.
import { appendFileSync } from 'node:fs';
import { createHash } from 'node:crypto';

export function imageDigests(value) {
  const found = [];
  function visit(item) {
    if (Array.isArray(item)) { for (const v of item) visit(v); return; }
    if (!item || typeof item !== 'object') return;
    const inline = item.inlineData || item.inline_data;
    if (inline && /^image\//.test(inline.mimeType || inline.mime_type || '')) {
      if (typeof inline.data !== 'string') throw new Error('Non-base64 image data');
      found.push(createHash('sha256').update(Buffer.from(inline.data, 'base64')).digest('hex'));
    }
    for (const [key, val] of Object.entries(item)) {
      if (key !== 'inlineData' && key !== 'inline_data') visit(val);
    }
  }
  visit(value); return found;
}

export function verifyPayload(body, expectedImage, expectedModel) {
  const hashes = imageDigests(body);
  const model = body.model || body.request?.model;
  if (model !== expectedModel) throw new Error('Model identity differs from frozen CLI model');
  if (hashes.length !== 1 || hashes[0] !== expectedImage) {
    throw new Error('Expected exactly one unchanged frozen PNG in the outgoing request');
  }
  return { request_model: model, image_sha256: hashes[0], image_count: 1 };
}

if (process.env.CLI_IMAGE_AUDIT_FILE) {
  const original = globalThis.fetch;
  if (typeof original !== 'function') throw new Error('Global fetch unavailable');
  globalThis.fetch = async function(input, init) {
    const url = new URL(typeof input === 'string' || input instanceof URL ? input : input.url);
    const isModel = /googleapis\.com$/.test(url.hostname) && /generateContent/i.test(url.pathname);
    if (!isModel) return original(input, init);
    const raw = init?.body ?? (input instanceof Request ? await input.clone().text() : undefined);
    if (typeof raw !== 'string') throw new Error('Unable to audit model request body');
    let evidence;
    try {
      evidence = verifyPayload(JSON.parse(raw), process.env.CLI_EXPECTED_IMAGE_SHA256,
                               process.env.CLI_EXPECTED_MODEL);
    } catch (error) {
      appendFileSync(process.env.CLI_IMAGE_AUDIT_FILE, JSON.stringify({ status:'blocked', reason:String(error) })+'\n');
      throw error;
    }
    appendFileSync(process.env.CLI_IMAGE_AUDIT_FILE, JSON.stringify({ status:'verified', ...evidence,
      at:new Date().toISOString(), host:url.hostname, path:url.pathname })+'\n');
    return original(input, init);
  };
}
