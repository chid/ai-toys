High value                                                                                                                                                           
  - Tool/function calling — pass a tools= list to chat(), handle the tool_calls response and loop back. Enables the agentic pattern. Both llamacpp server and Ollama
  support this.                                                                                                                                                        
  - Conversation class — thin wrapper that owns its own history list and calls self.llm.chat() automatically. Right now every example manages history manually; this
  would eliminate that boilerplate.                                                                                                                                    
  - Embeddings — embed(texts) -> list[list[float]] via /v1/embeddings. One method unlocks RAG, semantic search, clustering.                                            
                                         
  Moderate value                                                                                                                                                       
  - Async variant — achat() using AsyncOpenAI, so you can run multiple personas or dungeon events concurrently with asyncio.gather.
  - Token budget tracking — read usage from non-streaming responses, accumulate a session total, optionally warn when nearing the context limit. Useful for debugging  
  runaway reasoning models.                                                                                                                                          
  - Retry with backoff — wrap the API call in a loop with exponential backoff on 429/503. One-liner with tenacity or hand-rolled.                                      
                                                     
  Nice to have                                                                                                                                                         
  - Grammar-constrained JSON — for llamacpp, pass a GBNF grammar to force valid JSON output directly from the model rather than parsing after the fact. More reliable
  than prompt-engineering for structured output.                                                                                                                       
  - Structured output via response_format — for OpenAI-compat servers that support it, pass response_format={"type": "json_object"} or a JSON schema. ai_dungeon.py
  would benefit from this.                                                                                                                                             
  - Vision support — pass image URLs or base64 in the message content list. Single code path change in chat().                                                         
                                                                                                              
