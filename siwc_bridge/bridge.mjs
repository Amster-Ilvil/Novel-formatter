// Novel Formatter local-only IPC bridge for the official @siwc/local SDK.
// No OAuth tokens, refresh tokens, model prompts, or SDK errors are logged.
import { execFile, execFileSync } from 'node:child_process';
import { homedir, platform } from 'node:os';
import { readFile, stat } from 'node:fs/promises';
import { isAbsolute } from 'node:path';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { createInterface } from 'node:readline';

const base = join(homedir(), 'Library', 'Application Support', 'NovelFormatter');
const sdkRoot = join(base, 'siwc-devkit');
const sdkFile = join(sdkRoot, 'packages', 'local', 'dist', 'index.js');
const keychainTool = join(base, 'siwc-keychain-crypto');

function protectCredential() {
  if (platform() !== 'darwin') throw new Error('ChatGPT subscription sign-in requires local macOS Keychain runtime.');
  const call = (op, input) => execFileSync(keychainTool, [op], {
    input, timeout: 30000, maxBuffer: 16 * 1024 * 1024,
    stdio: ['pipe', 'pipe', 'ignore'],
  });
  return {
    id: 'nf-macos-keychain-cryptokit-v1',
    isAvailable: () => { try { call('check', Buffer.alloc(0)); return true; } catch { return false; } },
    encrypt: (plaintext) => Uint8Array.from(call('seal', Buffer.from(plaintext, 'utf8'))),
    decrypt: (ciphertext) => call('open', Buffer.from(ciphertext)).toString('utf8'),
  };
}

function safeError(err) {
  const code = typeof err?.code === 'string' && /^[a-z_0-9-]{1,70}$/i.test(err.code) ? err.code : 'siwc_error';
  // Never emit arbitrary exception strings: HTTP/SDK errors may contain user data.
  const allowed = new Set(['sign_in_required', 'sharing_not_enabled', 'cancelled', 'connection_busy',
    'reauth_required', 'rate_limit_exceeded', 'subscription_sharing_usage_limit_exceeded',
    'invalid_config', 'invalid_request', 'network_error', 'model_not_found', 'unsupported_model',
    'refresh_not_ready', 'credential_store_unavailable']);
  return { code, message: allowed.has(code) ? `ChatGPT sign-in request failed (${code}).` :
    'ChatGPT sign-in request failed. Check sign-in, model access, connection and subscription limits.' };
}

const lineReader = createInterface({ input: process.stdin, terminal: false, crlfDelay: Infinity });
let requestLine;
for await (const line of lineReader) { requestLine = line; break; }
let result;
try {
  if (!requestLine || requestLine.length > 18 * 1024 * 1024) throw new Error('missing or oversized command');
  const req = JSON.parse(requestLine);
  if (typeof req !== 'object' || !req || Array.isArray(req)) throw new Error('invalid command');
  const allowed = new Set(['session', 'login', 'logout', 'models', 'respond', 'profiles']);
  if (!allowed.has(req.action)) throw new Error('invalid operation');
  const { createChatGPT } = await import(pathToFileURL(sdkFile).href);
  const client = createChatGPT({
    appName: 'Novel Formatter', appId: 'novel-formatter', redirectPort: 0,
    storageDir: join(base, 'siwc-credentials'), credentialEncryption: protectCredential(),
    openBrowser: (url) => new Promise((resolve, reject) => {
      if (!url.startsWith('https://')) return reject(new Error('Invalid browser URL'));
      execFile('/usr/bin/open', [url], { timeout: 15000 }, (error) => error ? reject(error) : resolve());
    }),
  });
  let data;
  switch (req.action) {
    case 'session': data = await client.getSession(); break;
    case 'login': data = await client.signIn({ newProfile: Boolean(req.newProfile) }); break;
    case 'logout': await client.disconnect(); data = await client.getSession(); break;
    case 'profiles': data = await client.listProfiles(); break;
    case 'models': data = await client.listModels(); break;
    case 'respond': {
      if (typeof req.model !== 'string' || !/^[a-zA-Z0-9._-]{1,128}$/.test(req.model)) throw new Error('invalid model');
      if (typeof req.prompt !== 'string' || !req.prompt || req.prompt.length > 4000000) throw new Error('invalid input');
      // Image input is supported by the ChatGPT plan Responses route for eligible
      // models. The pinned @siwc/local SDK has a narrow, licensed validator
      // adaptation at install-time. The SDK alone owns token refresh + streaming.
      // Images stay local until the user explicitly requests model inference.
      const imagePaths = req.imagePaths ?? [];
      if (!Array.isArray(imagePaths) || imagePaths.length > 10) throw new Error('invalid image paths');
      let input = req.prompt;
      if (imagePaths.length) {
        const items = [{ type: 'input_text', text: req.prompt }];
        let total = 0;
        for (const path of imagePaths) {
          if (typeof path !== 'string' || !isAbsolute(path) || path.includes('\0') || path.length > 4096) throw new Error('invalid image path');
          const meta = await stat(path);
          if (!meta.isFile() || meta.size < 8 || meta.size > 12 * 1024 * 1024) throw new Error('invalid image size');
          total += meta.size;
          if (total > 30 * 1024 * 1024) throw new Error('images too large');
          const bytes = await readFile(path);
          let mime;
          if (bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))) mime = 'image/png';
          else if (bytes[0] === 255 && bytes[1] === 216 && bytes[2] === 255) mime = 'image/jpeg';
          else if (bytes.toString('ascii', 0, 4) === 'RIFF' && bytes.toString('ascii', 8, 12) === 'WEBP') mime = 'image/webp';
          else throw new Error('unsupported image file');
          items.push({ type: 'input_image', image_url: `data:${mime};base64,${bytes.toString('base64')}`, detail: 'high' });
        }
        input = [{ role: 'user', content: items }];
      }
      const reply = await client.streamResponse({ model: req.model, input });
      data = { text: reply.text };
      break;
    }
  }
  result = { ok: true, data };
} catch (error) {
  result = { ok: false, error: safeError(error) };
}
process.stdout.write(JSON.stringify(result) + '\n');
