css = """
<style>
.chat-bubble{padding:12px 16px;margin:8px 0;border-radius:12px;line-height:1.5}
.user{background:#1f6feb;color:#fff}
.bot{background:#111827;color:#e5e7eb;border:1px solid #374151}
</style>
"""

user_template = '<div class="chat-bubble user">{{MSG}}</div>'
bot_template = '<div class="chat-bubble bot">{{MSG}}</div>'