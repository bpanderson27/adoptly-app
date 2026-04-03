import os
import json
import anthropic
from flask import Flask, request, jsonify, Response, stream_with_context, render_template_string
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are a friendly restaurant guide assistant named MenuMind. Your job is to help users decide what to order at a specific restaurant.

When a user tells you a restaurant name:
1. Use the web_search tool to find the restaurant's current menu items with prices
2. Use the web_search tool to find recent reviews, popular dishes, and what people recommend

After gathering menu and review information, ask the user 2-3 targeted qualifying questions to understand their preferences. Ask all questions at once, numbered, such as:
- Any dietary restrictions or allergies?
- What are you in the mood for? (light/hearty/adventurous/comfort food/etc.)
- Any budget preference for the dish?

Once you have their answers, provide 3-5 specific menu recommendations formatted clearly, each with:
- **Dish Name** - price (if found)
- Why it matches their preferences
- What reviewers say about it (if available)

Be warm, enthusiastic, and concise. Use markdown formatting for your final recommendations."""

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MenuMind - Restaurant Recommender</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
      background: #0f0f0f;
      color: #e8e8e8;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }

    header {
      background: #1a1a1a;
      border-bottom: 1px solid #2a2a2a;
      padding: 16px 24px;
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .logo { font-size: 22px; font-weight: 700; color: #f97316; }
    .tagline { font-size: 13px; color: #888; }

    .chat-container {
      flex: 1;
      display: flex;
      flex-direction: column;
      max-width: 800px;
      width: 100%;
      margin: 0 auto;
      padding: 24px 16px 0;
      gap: 16px;
    }

    .welcome-card {
      background: #1a1a1a;
      border: 1px solid #2a2a2a;
      border-radius: 16px;
      padding: 32px;
      text-align: center;
    }

    .welcome-card h2 { font-size: 24px; font-weight: 700; color: #f97316; margin-bottom: 8px; }
    .welcome-card p { color: #aaa; font-size: 15px; line-height: 1.6; margin-bottom: 24px; }

    .restaurant-input-group { display: flex; gap: 10px; }

    .restaurant-input-group input {
      flex: 1;
      background: #0f0f0f;
      border: 1px solid #333;
      border-radius: 10px;
      padding: 12px 16px;
      color: #e8e8e8;
      font-size: 15px;
      outline: none;
      transition: border-color 0.2s;
    }

    .restaurant-input-group input:focus { border-color: #f97316; }
    .restaurant-input-group input::placeholder { color: #555; }

    .btn-start {
      background: #f97316;
      color: #fff;
      border: none;
      border-radius: 10px;
      padding: 12px 20px;
      font-size: 15px;
      font-weight: 600;
      cursor: pointer;
      white-space: nowrap;
      transition: background 0.2s;
    }

    .btn-start:hover { background: #ea6a0a; }
    .btn-start:disabled { background: #555; cursor: not-allowed; }

    #messages { display: flex; flex-direction: column; gap: 16px; }

    .message { display: flex; gap: 12px; align-items: flex-start; }
    .message.user { flex-direction: row-reverse; }

    .avatar {
      width: 36px;
      height: 36px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 18px;
      flex-shrink: 0;
    }

    .message.assistant .avatar { background: #f97316; }
    .message.user .avatar { background: #3a3a3a; }

    .bubble {
      max-width: 85%;
      padding: 14px 18px;
      border-radius: 16px;
      font-size: 15px;
      line-height: 1.65;
    }

    .message.assistant .bubble {
      background: #1a1a1a;
      border: 1px solid #2a2a2a;
      border-top-left-radius: 4px;
    }

    .message.user .bubble {
      background: #f97316;
      color: #fff;
      border-top-right-radius: 4px;
    }

    .bubble strong { color: #f97316; }
    .message.user .bubble strong { color: #fff; }
    .bubble h3 { font-size: 16px; margin: 12px 0 6px; color: #f97316; }
    .message.user .bubble h3 { color: #fff; }
    .bubble ul, .bubble ol { padding-left: 20px; margin: 8px 0; }
    .bubble li { margin: 4px 0; }
    .bubble hr { border: none; border-top: 1px solid #2a2a2a; margin: 12px 0; }
    .bubble p { margin: 6px 0; }

    .typing-indicator {
      display: flex;
      gap: 5px;
      align-items: center;
      padding: 16px 18px;
    }

    .typing-indicator span {
      width: 8px;
      height: 8px;
      background: #555;
      border-radius: 50%;
      animation: bounce 1.2s infinite;
    }

    .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
    .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }

    @keyframes bounce {
      0%, 60%, 100% { transform: translateY(0); }
      30% { transform: translateY(-8px); }
    }

    .searching-badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      background: #0f2010;
      border: 1px solid #1a4020;
      border-radius: 20px;
      padding: 6px 14px;
      font-size: 13px;
      color: #4ade80;
    }

    .searching-badge .dot {
      width: 7px;
      height: 7px;
      background: #4ade80;
      border-radius: 50%;
      animation: pulse 1s infinite;
    }

    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.4; transform: scale(0.8); }
    }

    .search-query {
      font-size: 12px;
      color: #666;
      margin-top: 6px;
      font-style: italic;
    }

    .input-area {
      position: sticky;
      bottom: 0;
      background: #0f0f0f;
      padding: 16px 0 20px;
      border-top: 1px solid #1a1a1a;
      margin-top: 16px;
    }

    .input-row { display: flex; gap: 10px; align-items: flex-end; }

    #user-input {
      flex: 1;
      background: #1a1a1a;
      border: 1px solid #333;
      border-radius: 12px;
      padding: 12px 16px;
      color: #e8e8e8;
      font-size: 15px;
      font-family: inherit;
      resize: none;
      outline: none;
      max-height: 120px;
      min-height: 48px;
      transition: border-color 0.2s;
    }

    #user-input:focus { border-color: #f97316; }
    #user-input::placeholder { color: #555; }

    .btn-send {
      background: #f97316;
      border: none;
      border-radius: 12px;
      width: 48px;
      height: 48px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: background 0.2s;
      flex-shrink: 0;
    }

    .btn-send:hover { background: #ea6a0a; }
    .btn-send:disabled { background: #444; cursor: not-allowed; }
    .btn-send svg { width: 20px; height: 20px; fill: #fff; }

    .hidden { display: none !important; }
  </style>
</head>
<body>
  <header>
    <div>
      <div class="logo">🍽️ MenuMind</div>
      <div class="tagline">AI-powered restaurant recommendations</div>
    </div>
  </header>

  <div class="chat-container">
    <div id="welcome-card" class="welcome-card">
      <h2>What are you in the mood for?</h2>
      <p>Enter a restaurant name and I'll pull up the menu and reviews,<br>then help you find the perfect dish for your mood.</p>
      <div class="restaurant-input-group">
        <input type="text" id="restaurant-input" placeholder="e.g. The French Laundry, Shake Shack NYC, Nobu Malibu..." />
        <button class="btn-start" id="start-btn" onclick="startSession()">Let's Go →</button>
      </div>
    </div>

    <div id="messages"></div>

    <div id="input-area" class="input-area hidden">
      <div class="input-row">
        <textarea id="user-input" placeholder="Type your answer..." rows="1"
          onkeydown="handleKey(event)" oninput="autoResize(this)"></textarea>
        <button class="btn-send" id="send-btn" onclick="sendMessage()">
          <svg viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
        </button>
      </div>
    </div>
  </div>

  <script>
    let conversationHistory = [];
    let isStreaming = false;

    function startSession() {
      const input = document.getElementById('restaurant-input');
      const restaurant = input.value.trim();
      if (!restaurant) { input.focus(); return; }

      document.getElementById('welcome-card').classList.add('hidden');
      document.getElementById('input-area').classList.remove('hidden');

      const userMsg = `I want to go to ${restaurant}`;
      conversationHistory.push({ role: 'user', content: userMsg });
      addMessage('user', userMsg);
      streamResponse();
    }

    function handleKey(e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    }

    function autoResize(el) {
      el.style.height = 'auto';
      el.style.height = Math.min(el.scrollHeight, 120) + 'px';
    }

    function sendMessage() {
      if (isStreaming) return;
      const input = document.getElementById('user-input');
      const text = input.value.trim();
      if (!text) return;
      input.value = '';
      input.style.height = '';
      conversationHistory.push({ role: 'user', content: text });
      addMessage('user', text);
      streamResponse();
    }

    function addMessage(role, text) {
      const messages = document.getElementById('messages');
      const div = document.createElement('div');
      div.className = `message ${role}`;
      const avatar = document.createElement('div');
      avatar.className = 'avatar';
      avatar.textContent = role === 'assistant' ? '🍽️' : '👤';
      const bubble = document.createElement('div');
      bubble.className = 'bubble';
      bubble.innerHTML = renderMarkdown(text);
      div.appendChild(avatar);
      div.appendChild(bubble);
      messages.appendChild(div);
      scrollDown();
      return bubble;
    }

    function renderMarkdown(text) {
      return text
        .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/^### (.+)$/gm, '<h3>$1</h3>')
        .replace(/^## (.+)$/gm, '<h3>$1</h3>')
        .replace(/^(\d+)\. (.+)$/gm, '<li>$2</li>')
        .replace(/^[-*] (.+)$/gm, '<li>$1</li>')
        .replace(/(<li>[\s\S]+?<\/li>)+/g, m => '<ul>' + m + '</ul>')
        .replace(/---/g, '<hr>')
        .replace(/\n\n/g, '<br><br>')
        .replace(/\n/g, '<br>');
    }

    function scrollDown() {
      setTimeout(() => window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' }), 50);
    }

    async function streamResponse() {
      isStreaming = true;
      document.getElementById('send-btn').disabled = true;

      const messagesEl = document.getElementById('messages');
      let thinkingEl = createThinkingEl();
      messagesEl.appendChild(thinkingEl);
      scrollDown();

      let assistantBubble = null;
      let fullText = '';
      let searchingEl = null;

      try {
        const resp = await fetch('/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ messages: conversationHistory })
        });

        if (!resp.ok) throw new Error(`Server error ${resp.status}`);

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });

          const lines = buffer.split('\n');
          buffer = lines.pop();

          for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            const raw = line.slice(6).trim();
            if (!raw || raw === '[DONE]') continue;

            let event;
            try { event = JSON.parse(raw); } catch { continue; }

            if (event.type === 'searching') {
              thinkingEl?.remove(); thinkingEl = null;
              if (!searchingEl) {
                searchingEl = createSearchingEl(event.query);
                messagesEl.appendChild(searchingEl);
                scrollDown();
              } else {
                const q = searchingEl.querySelector('.search-query');
                if (q && event.query) q.textContent = `"${event.query}"`;
              }
            }

            if (event.type === 'text') {
              thinkingEl?.remove(); thinkingEl = null;
              searchingEl?.remove(); searchingEl = null;

              if (!assistantBubble) {
                const msgDiv = document.createElement('div');
                msgDiv.className = 'message assistant';
                const av = document.createElement('div');
                av.className = 'avatar';
                av.textContent = '🍽️';
                assistantBubble = document.createElement('div');
                assistantBubble.className = 'bubble';
                msgDiv.appendChild(av);
                msgDiv.appendChild(assistantBubble);
                messagesEl.appendChild(msgDiv);
              }

              fullText += event.delta;
              assistantBubble.innerHTML = renderMarkdown(fullText);
              scrollDown();
            }
          }
        }

      } catch (e) {
        thinkingEl?.remove();
        searchingEl?.remove();
        addMessage('assistant', `Sorry, something went wrong: ${e.message}`);
      }

      thinkingEl?.remove();
      searchingEl?.remove();

      if (fullText) conversationHistory.push({ role: 'assistant', content: fullText });

      isStreaming = false;
      document.getElementById('send-btn').disabled = false;
      document.getElementById('user-input').focus();
    }

    function createThinkingEl() {
      const div = document.createElement('div');
      div.className = 'message assistant';
      div.innerHTML = `<div class="avatar">🍽️</div>
        <div class="bubble typing-indicator"><span></span><span></span><span></span></div>`;
      return div;
    }

    function createSearchingEl(query) {
      const div = document.createElement('div');
      div.className = 'message assistant';
      div.innerHTML = `<div class="avatar">🍽️</div>
        <div class="bubble">
          <div class="searching-badge"><div class="dot"></div>Searching the web...</div>
          ${query ? `<div class="search-query">"${query}"</div>` : ''}
        </div>`;
      return div;
    }

    document.getElementById('restaurant-input').addEventListener('keydown', e => {
      if (e.key === 'Enter') startSession();
    });
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    messages = data.get("messages", [])

    def generate():
        tools = [{"type": "web_search_20260209", "name": "web_search"}]
        current_messages = list(messages)
        max_continuations = 5

        for _ in range(max_continuations):
            try:
                with client.messages.stream(
                    model="claude-opus-4-6",
                    max_tokens=4096,
                    system=SYSTEM_PROMPT,
                    tools=tools,
                    messages=current_messages,
                ) as stream:
                    response_content = []
                    stop_reason = None

                    for event in stream:
                        etype = event.type

                        if etype == "content_block_start":
                            block = event.content_block
                            if block.type == "server_tool_use":
                                # Web search starting
                                query = getattr(block, 'input', {})
                                if isinstance(query, dict):
                                    q = query.get("query", "")
                                else:
                                    q = ""
                                yield f"data: {json.dumps({'type': 'searching', 'query': q})}\n\n"
                            response_content.append({"type": block.type, "_raw": block})

                        elif etype == "content_block_delta":
                            delta = event.delta
                            if delta.type == "text_delta":
                                yield f"data: {json.dumps({'type': 'text', 'delta': delta.text})}\n\n"
                            elif delta.type == "input_json_delta":
                                # Building search query - emit updated query if possible
                                pass

                        elif etype == "message_delta":
                            stop_reason = event.delta.stop_reason

                    # Get the final message to check stop_reason
                    final = stream.get_final_message()
                    stop_reason = final.stop_reason

                    if stop_reason == "pause_turn":
                        # Server-side tool hit its iteration limit; re-send to continue
                        current_messages.append({
                            "role": "assistant",
                            "content": final.content
                        })
                        continue
                    else:
                        # end_turn or other terminal state
                        break

            except anthropic.APIError as e:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                break

        yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
