'use strict';
const fs = require('node:fs');

function atomicJson(file, value) {
  const temporary = `${file}.tmp-${process.pid}`;
  fs.writeFileSync(temporary, JSON.stringify(value, null, 2));
  fs.renameSync(temporary, file);
}
function safeError(error) {
  return String(error.stack || error.message || error).replace(/https?:\/\/\S+/g, '[url]');
}
function terminal(error) {
  return /Target page, context or browser has been closed|Owned target missing|Refusing overwrite|PAYLOAD_CONFLICT|AUTH_REQUIRED/.test(String(error.message));
}
async function retryOperation(action, {attempts = 3, onAttempt = () => {}, wait = ms => new Promise(r => setTimeout(r, ms))} = {}) {
  for (let attempt = 1; attempt <= attempts; attempt++) {
    try {
      const value = await action(attempt);
      onAttempt({attempt, status: 'done'});
      return value;
    } catch (error) {
      const isTerminal = terminal(error);
      onAttempt({attempt, status: 'failed', terminal: isTerminal, error: safeError(error)});
      if (isTerminal || attempt === attempts) throw error;
      await wait(2000 * (2 ** (attempt - 1)));
    }
  }
}
function preservePayload(source, destination) {
  if (fs.existsSync(destination)) {
    if (!fs.readFileSync(source).equals(fs.readFileSync(destination))) throw Error('PAYLOAD_CONFLICT: existing payload differs; preserved without overwrite');
    return;
  }
  fs.copyFileSync(source, destination, fs.constants.COPYFILE_EXCL);
}
module.exports = {atomicJson, safeError, terminal, retryOperation, preservePayload};
