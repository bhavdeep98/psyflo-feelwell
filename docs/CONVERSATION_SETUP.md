# Conversation Agent Setup Guide

Quick guide to set up the conversational AI agent.

## Prerequisites

- Python 3.11+
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

## Step 1: Install Dependencies

```bash
# Install LangChain and related packages
pip install langchain langchain-openai langchain-core langgraph python-dotenv

# Or install all requirements
pip install -r requirements.txt
```

## Step 2: Configure API Key

### Option A: Using .env file (Recommended)

1. Copy the example file:
```bash
cp backend/.env.example backend/.env
```

2. Edit `backend/.env` and add your OpenAI API key:
```bash
# backend/.env

# OpenAI API Configuration
OPENAI_API_KEY=sk-your-actual-key-here
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.7
```

### Option B: Environment Variable

```bash
export OPENAI_API_KEY=sk-your-actual-key-here
```

## Step 3: Test the Agent

```bash
# Run conversation agent tests
pytest tests/test_conversation_agent.py -v

# Expected output:
# ✓ test_initialization_fails_without_api_key
# ✓ test_initialization_succeeds_with_api_key
# ✓ test_crisis_override_response
# ... (more tests)
```

## Step 4: Start the Backend

```bash
cd backend
uvicorn main:app --reload
```

The backend will:
1. Load `.env` file
2. Initialize conversation agent
3. Fail loud if `OPENAI_API_KEY` is missing

## Step 5: Test End-to-End

1. Start frontend:
```bash
cd frontend
npm run dev
```

2. Open http://localhost:5173

3. Chat as a student - you should now get empathetic AI responses!

## Troubleshooting

### Error: "OPENAI_API_KEY not found"

**Cause:** API key not set in `.env` file or environment.

**Fix:**
```bash
# Check if .env exists
ls backend/.env

# If not, create it
cp backend/.env.example backend/.env

# Edit and add your key
nano backend/.env
```

### Error: "Failed to initialize LLM"

**Cause:** Invalid API key or network issue.

**Fix:**
1. Verify your API key at https://platform.openai.com/api-keys
2. Check internet connection
3. Verify OpenAI API status: https://status.openai.com/

### Error: "Module 'langchain' not found"

**Cause:** Dependencies not installed.

**Fix:**
```bash
pip install langchain langchain-openai langchain-core langgraph
```

## Configuration Options

### Model Selection

Use different OpenAI models in `.env`:

```bash
# Faster, cheaper (default)
OPENAI_MODEL=gpt-4o-mini

# More capable, slower
OPENAI_MODEL=gpt-4o

# Most capable
OPENAI_MODEL=gpt-4-turbo
```

### Temperature

Control response creativity (0.0 = deterministic, 1.0 = creative):

```bash
# More consistent (recommended for mental health)
OPENAI_TEMPERATURE=0.7

# More creative
OPENAI_TEMPERATURE=0.9

# More deterministic
OPENAI_TEMPERATURE=0.5
```

## Cost Estimation

**gpt-4o-mini pricing (as of Jan 2026):**
- Input: $0.15 per 1M tokens
- Output: $0.60 per 1M tokens

**Typical conversation:**
- Student message: ~50 tokens
- AI response: ~100 tokens
- Cost per message: ~$0.00015 (0.015 cents)

**Monthly estimate (1000 students, 10 messages/day):**
- Total messages: 300,000/month
- Estimated cost: ~$45/month

## Security Best Practices

### ✅ DO:
- Store API key in `.env` file
- Add `.env` to `.gitignore`
- Use separate keys for dev/staging/prod
- Rotate keys regularly
- Monitor usage at https://platform.openai.com/usage

### ❌ DON'T:
- Commit `.env` to git
- Share API keys in Slack/email
- Use production keys in development
- Hard-code API keys in source code

## Next Steps

1. ✅ Set up API key
2. ✅ Test conversation agent
3. ✅ Start backend and frontend
4. 📝 Customize system prompts (see `src/conversation/conversation_agent.py`)
5. 📊 Monitor usage and costs
6. 🚀 Deploy to production

## Support

- **LangChain Docs:** https://python.langchain.com/
- **OpenAI API Docs:** https://platform.openai.com/docs/
- **Project Issues:** [GitHub Issues](https://github.com/your-repo/issues)
