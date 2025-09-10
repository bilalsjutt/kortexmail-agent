from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document
from langchain_community.vectorstores import UpstashVectorStore
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langsmith import traceable

from upstash_redis import Redis
import os
from dotenv import load_dotenv
import logging
from datetime import datetime
import asyncio
import uuid
import json
import tiktoken

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OPTIMIZED: Lighter LLM for memory operations
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.1, max_tokens=200)


class MemoryManager:
    def __init__(self, window_size: int = 5, max_tokens: int = 1500):  # OPTIMIZED: Reduced defaults
        self.window_size = window_size
        self.max_tokens = max_tokens

        # OPTIMIZED: Simpler tokenizer initialization
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # OPTIMIZED: Fail fast on missing credentials
        self.redis_url = os.getenv("UPSTASH_REDIS_REST_URL")
        self.redis_token = os.getenv("UPSTASH_REDIS_REST_TOKEN")

        if not self.redis_url or not self.redis_token:
            logger.warning("Redis credentials not found, memory features disabled")
            self.redis = None
            self.vector_store = None
            self.retriever = None
            return

        try:
            self.redis = Redis(url=self.redis_url, token=self.redis_token)

            # OPTIMIZED: Faster Redis connection test
            asyncio.create_task(self._test_redis_connection())

            # OPTIMIZED: Optional vector store initialization
            vector_url = os.getenv("UPSTASH_VECTOR_REST_URL")
            vector_token = os.getenv("UPSTASH_VECTOR_REST_TOKEN")

            if vector_url and vector_token:
                self.embedding = OpenAIEmbeddings(model='text-embedding-3-small')
                self.vector_store = UpstashVectorStore(
                    embedding=self.embedding,
                    index_url=vector_url,
                    index_token=vector_token,
                    text_key="content"
                )

                # OPTIMIZED: Simpler retriever setup
                compressor = LLMChainExtractor.from_llm(llm)
                self.retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=self.vector_store.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 2}  # OPTIMIZED: Reduced from 3 to 2
                    )
                )
            else:
                logger.warning("Vector store credentials missing, using Redis only")
                self.vector_store = None
                self.retriever = None

        except Exception as e:
            logger.error(f"Memory manager initialization error: {e}")
            self.redis = None
            self.vector_store = None
            self.retriever = None

    async def _test_redis_connection(self):
        """OPTIMIZED: Faster Redis connection test"""
        try:
            if self.redis:
                await asyncio.wait_for(
                    asyncio.to_thread(self.redis.ping),
                    timeout=2  # OPTIMIZED: Reduced timeout
                )
                logger.info("Redis connection successful")
        except Exception as e:
            logger.warning(f"Redis connection test failed: {e}")
            self.redis = None

    def get_session_key(self, session_id: str) -> str:
        """Get Redis key for session"""
        return f"chat_session_{session_id}"

    def _count_tokens(self, text: str) -> int:
        """OPTIMIZED: Faster token counting with caching"""
        if not text:
            return 0
        try:
            return len(self.tokenizer.encode(text))
        except Exception:
            # OPTIMIZED: More accurate fallback
            return len(text) // 3

    @traceable(name="add_conversation_to_memory")
    async def add_conversation_to_memory(self, session_id: str, human_msg: str, ai_msg: str):
        """OPTIMIZED: Faster memory storage with minimal processing"""
        if not self.redis:
            return

        # OPTIMIZED: Skip processing for very short exchanges
        if len(human_msg.strip()) < 5 or len(ai_msg.strip()) < 10:
            return

        conversation_text = f"Human: {human_msg}\nAI: {ai_msg}"
        token_count = self._count_tokens(conversation_text)

        # OPTIMIZED: Higher threshold for summarization
        if token_count > self.max_tokens:
            try:
                summary_data = await asyncio.wait_for(
                    self._summarize_conversation(human_msg, ai_msg),
                    timeout=5  # OPTIMIZED: Timeout for summarization
                )
                storage_human = summary_data["human"]
                storage_ai = summary_data["ai"]
                is_summary = True
            except asyncio.TimeoutError:
                logger.warning("Summarization timed out, using truncated text")
                storage_human = self._smart_truncate(human_msg, 200)
                storage_ai = self._smart_truncate(ai_msg, 300)
                is_summary = True
        else:
            storage_human = human_msg
            storage_ai = ai_msg
            is_summary = False

        conversation_data = {
            "id": str(uuid.uuid4())[:8],
            "human": storage_human,
            "ai": storage_ai,
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "is_summary": is_summary,
            "token_count": self._count_tokens(f"{storage_human}\n{storage_ai}")
        }

        try:
            # OPTIMIZED: Redis-only storage for speed, vector store optional
            await asyncio.wait_for(
                self._store_in_redis(session_id, conversation_data),
                timeout=3
            )

            # OPTIMIZED: Background vector storage for substantial exchanges only
            if (self.vector_store and not is_summary and
                    self._is_substantial_exchange(human_msg, ai_msg)):
                asyncio.create_task(self._store_in_vector_store({
                    **conversation_data,
                    "human": human_msg,
                    "ai": ai_msg
                }))

            logger.debug(f"Stored conversation for session {session_id}")

        except Exception as e:
            logger.error(f"Error storing conversation: {e}")

    async def _store_in_redis(self, session_id: str, conversation_data: dict):
        """OPTIMIZED: Faster Redis operations"""
        session_key = self.get_session_key(session_id)
        conversation_json = json.dumps(conversation_data)

        # OPTIMIZED: Single pipeline for all operations
        try:
            await asyncio.to_thread(self._redis_pipeline_operations,
                                    session_key, conversation_json)
        except Exception as e:
            logger.error(f"Redis storage error: {e}")
            raise

    def _redis_pipeline_operations(self, session_key: str, conversation_json: str):
        """OPTIMIZED: Execute Redis operations without pipeline (Upstash doesn't support pipelines)"""
        try:
            # OPTIMIZED: Upstash Redis doesn't support pipelines, so execute commands individually
            self.redis.lpush(session_key, conversation_json)
            self.redis.ltrim(session_key, 0, self.window_size - 1)
            self.redis.expire(session_key, 86400)  # 24 hours
        except Exception as e:
            logger.error(f"Redis operations failed: {e}")
            raise

    async def _store_in_vector_store(self, conversation_data: dict):
        """OPTIMIZED: Background vector storage"""
        try:
            conversation_text = f"Human: {conversation_data['human']}\nAI: {conversation_data['ai']}"

            document = Document(
                page_content=conversation_text,
                metadata={
                    "conversation_id": conversation_data["id"],
                    "timestamp": conversation_data["timestamp"],
                    "session_id": conversation_data["session_id"],
                    "type": "conversation_exchange",
                    "token_count": conversation_data.get("token_count", 0)
                }
            )

            await asyncio.wait_for(
                asyncio.to_thread(self.vector_store.add_documents, [document]),
                timeout=10
            )
            logger.debug(f"Stored conversation {conversation_data['id']} in vector store")

        except Exception as e:
            logger.warning(f"Vector store error (non-critical): {e}")

    def _is_substantial_exchange(self, human_msg: str, ai_msg: str) -> bool:
        """OPTIMIZED: Quick check for substantial content"""
        if len(human_msg.strip()) < 15 or len(ai_msg.strip()) < 30:
            return False

        # OPTIMIZED: Simple word count check
        return len(human_msg.split()) > 5 and len(ai_msg.split()) > 10

    async def _summarize_conversation(self, human_msg: str, ai_msg: str) -> dict:
        """OPTIMIZED: Faster conversation summarization"""
        try:
            conversation_text = f"Human: {human_msg}\nAI: {ai_msg}"

            # OPTIMIZED: Shorter, more direct prompt
            summary_prompt = f"""Summarize this conversation briefly (under 100 words):

{conversation_text}

Format: Human: [key points] | AI: [key response]"""

            response = await asyncio.to_thread(llm.invoke, summary_prompt)
            summary = response.content.strip()

            # OPTIMIZED: Simple parsing
            if " | " in summary:
                parts = summary.split(" | ", 1)
                human_summary = parts[0].replace("Human:", "").strip()
                ai_summary = parts[1].replace("AI:", "").strip()
            else:
                # Fallback
                human_summary = self._smart_truncate(human_msg, 150)
                ai_summary = self._smart_truncate(summary, 250)

            return {
                "human": human_summary,
                "ai": ai_summary,
                "is_summary": True,
                "original_tokens": self._count_tokens(conversation_text)
            }

        except Exception as e:
            logger.error(f"Summarization error: {e}")
            return {
                "human": self._smart_truncate(human_msg, 150),
                "ai": self._smart_truncate(ai_msg, 250),
                "is_summary": True,
                "original_tokens": self._count_tokens(f"{human_msg}\n{ai_msg}")
            }

    def _smart_truncate(self, text: str, max_length: int) -> str:
        """OPTIMIZED: Faster text truncation"""
        if len(text) <= max_length:
            return text

        # OPTIMIZED: Simple truncation with word boundary
        truncated = text[:max_length - 3]
        last_space = truncated.rfind(' ')
        if last_space > max_length * 0.7:  # Keep if we don't lose too much
            truncated = truncated[:last_space]

        return truncated + "..."

    @traceable(name="get_session_memory_context")
    async def get_session_memory_context(self, session_id: str, user_message: str = "", llm_instance=None) -> str:
        """OPTIMIZED: Faster memory context retrieval"""
        if not self.redis:
            return ""

        try:
            session_key = self.get_session_key(session_id)

            # OPTIMIZED: Shorter timeout for Redis operations
            conversations_json = await asyncio.wait_for(
                asyncio.to_thread(self.redis.lrange, session_key, 0, -1),
                timeout=1
            )

            if not conversations_json:
                return ""

            # OPTIMIZED: Simpler context building
            context_parts = [f"[Recent conversations - Session {session_id}]:"]

            for i, conv_json in enumerate(conversations_json[:5]):  # OPTIMIZED: Limit to 5 recent
                try:
                    conv_data = json.loads(conv_json)
                    summary_marker = " [SUMM]" if conv_data.get("is_summary") else ""
                    context_parts.append(f"\n{i + 1}.{summary_marker} Human: {conv_data['human']}")
                    context_parts.append(f"   AI: {conv_data['ai']}")
                except json.JSONDecodeError:
                    continue

            session_context = "\n".join(context_parts)

            # OPTIMIZED: Skip vector search for simple requests
            if (user_message and llm_instance and len(user_message.split()) > 8 and
                    await self._should_use_vector_search_simple(user_message)):
                try:
                    vector_context = await asyncio.wait_for(
                        self.search_vector_store(user_message, session_id),
                        timeout=3
                    )
                    if vector_context:
                        return f"{session_context}\n\n{vector_context}"
                except asyncio.TimeoutError:
                    logger.debug("Vector search timed out, using session context only")

            return session_context

        except Exception as e:
            logger.error(f"Memory context error: {e}")
            return ""

    async def _should_use_vector_search_simple(self, user_input: str) -> bool:
        """OPTIMIZED: Simple heuristic for vector search decision"""
        if not self.retriever:
            return False

        # OPTIMIZED: Simple keyword-based decision
        search_indicators = [
            'before', 'previous', 'earlier', 'discussed', 'mentioned',
            'talked about', 'said', 'remember', 'recall'
        ]

        return any(indicator in user_input.lower() for indicator in search_indicators)

    @traceable(name="search_vector_store")
    async def search_vector_store(self, query: str, session_id: str = None) -> str:
        """OPTIMIZED: Faster vector search"""
        if not self.retriever:
            return ""

        try:
            docs = await asyncio.wait_for(
                asyncio.to_thread(self.retriever.get_relevant_documents, query),
                timeout=5  # OPTIMIZED: Reduced timeout
            )

            if not docs:
                return ""

            # OPTIMIZED: Filter and format more efficiently
            if session_id:
                docs = [doc for doc in docs if doc.metadata.get('session_id') == session_id]

            if not docs:
                return ""

            # OPTIMIZED: Simpler formatting
            context_parts = ["[Relevant past conversations]:"]
            for i, doc in enumerate(docs[:2], 1):  # OPTIMIZED: Limit to 2 results
                timestamp = doc.metadata.get('timestamp', 'unknown')
                summary_marker = " [SUMM]" if doc.metadata.get('is_summary') else ""
                context_parts.append(f"\n{i}.{summary_marker} [{timestamp}]")
                context_parts.append(f"   {doc.page_content}")

            return "\n".join(context_parts)

        except Exception as e:
            logger.warning(f"Vector search error: {e}")
            return ""

    async def get_memory_stats(self, session_id: str) -> dict:
        """OPTIMIZED: Faster memory stats"""
        stats = {
            "redis_available": self.redis is not None,
            "vector_store_available": self.vector_store is not None,
            "session_conversations": 0,
            "total_tokens": 0,
            "summaries_count": 0
        }

        if not self.redis:
            return stats

        try:
            session_key = self.get_session_key(session_id)
            conversations_json = await asyncio.wait_for(
                asyncio.to_thread(self.redis.lrange, session_key, 0, -1),
                timeout=1
            )

            stats["session_conversations"] = len(conversations_json)

            # OPTIMIZED: Process only if we have conversations
            if conversations_json:
                for conv_json in conversations_json:
                    try:
                        conv_data = json.loads(conv_json)
                        stats["total_tokens"] += conv_data.get("token_count", 0)
                        if conv_data.get("is_summary", False):
                            stats["summaries_count"] += 1
                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            logger.error(f"Memory stats error: {e}")

        return stats