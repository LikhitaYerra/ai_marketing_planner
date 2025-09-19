import os
from openai import OpenAI
from newsapi import NewsApiClient
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
import streamlit as st
import json
from datetime import datetime, timedelta
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.date import DateTrigger
import requests
import base64
from streamlit_oauth import OAuth2Component, StreamlitOauthError
from typing import List, Optional
import logging


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Initialize session state
if 'feedback_data' not in st.session_state:
    st.session_state.feedback_data = []
if 'scheduled_articles' not in st.session_state:
    # Load scheduled articles from persistent storage on startup
    try:
        file_path = os.path.abspath('articles_data.json')
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            with open(file_path, 'r') as f:
                content = f.read()
                if content.strip():
                    try:
                        articles = json.loads(content)
                        st.session_state.scheduled_articles = articles
                        print(f"[STARTUP] Loaded {len(articles)} articles from persistent storage")
                    except json.JSONDecodeError:
                        st.session_state.scheduled_articles = []
                        print("[STARTUP] Error parsing articles_data.json, initializing empty list")
                else:
                    st.session_state.scheduled_articles = []
                    print("[STARTUP] Empty articles_data.json file, initializing empty list")
        else:
            st.session_state.scheduled_articles = []
            print("[STARTUP] No articles_data.json file found, initializing empty list")
    except Exception as e:
        st.session_state.scheduled_articles = []
        print(f"[STARTUP] Error loading articles: {str(e)}")
        import traceback
        print(traceback.format_exc())

# Initialize scheduler
scheduler = BackgroundScheduler()
scheduler.start()

# Initialize API clients using environment variables
def init_api_clients():
    openai_key = st.secrets["OPENAI_API_KEY"]
    newsapi_key = st.secrets["NEWSAPI_KEY"]
    if not openai_key:
        st.error("OpenAI API key not found. Please check your .env file.")
        return None, None
    
    if not newsapi_key:
        st.error("News API key not found. Please check your .env file.")
        return None, None
    
    try:
        openai_client = OpenAI(api_key=openai_key)
        newsapi_client = NewsApiClient(api_key=newsapi_key)
        return openai_client, newsapi_client
    except Exception as e:
        st.error(f"Error initializing API clients: {str(e)}")
        return None, None

# Initialize clients
client, newsapi = init_api_clients()

# Check if clients are initialized
if not client or not newsapi:
    st.error("Cannot proceed without valid API keys. Please check your configuration.")
    st.stop()

# MCPGoogleNewsConnector class with enhanced filtering
class MCPGoogleNewsConnector:
    def __init__(self, api_client):
        self.api_client = api_client

    def fetch_headlines(self, keywords, max_results=5, language='en', sort_by='relevancy', from_date=None):
        headlines = []
        for keyword in keywords:
            articles = self.api_client.get_everything(
                q=keyword,
                language=language,
                page_size=max_results,
                sort_by=sort_by,
                from_param=from_date
            )
            for article in articles['articles']:
                headlines.append({
                    'title': article['title'],
                    'source': article['source']['name'],
                    'date': article['publishedAt']
                })
        return list(set([(h['title'], h['source'], h['date']) for h in headlines]))[:10]

# RAG Component with knowledge base management
class RAGManager:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = CharacterTextSplitter(chunk_size=200)
        self.vectorstore = None
        self.load_knowledge_base()

    def load_knowledge_base(self):
        try:
            self.vectorstore = FAISS.load_local("knowledge_base", self.embeddings)
        except:
            self.vectorstore = self.create_initial_knowledge_base()

    def create_initial_knowledge_base(self):
        initial_guidelines = '''
        Our content tone is professional yet accessible.
        Use actionable tips and lists in blog posts.
        Avoid technical jargon unless targeting developers.
        Prefer posts under 1200 words for better engagement.
        Include strong call-to-actions.
        '''
        docs = [Document(page_content=chunk) for chunk in self.text_splitter.split_text(initial_guidelines)]
        return FAISS.from_documents(docs, self.embeddings)

    def add_to_knowledge_base(self, new_text):
        docs = [Document(page_content=chunk) for chunk in self.text_splitter.split_text(new_text)]
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        else:
            self.vectorstore.add_documents(docs)
        self.vectorstore.save_local("knowledge_base")

    def get_retriever(self):
        return self.vectorstore.as_retriever()


class MarketingPipeline:
    """Unified MVP pipeline for ingesting SEO reports and generating content."""

    SUPPORTED_LANGUAGES = {
        "English": "en",
        "French": "fr",
    }

    def __init__(self, openai_client: OpenAI, rag_manager: RAGManager):
        self.client = openai_client
        self.rag_manager = rag_manager
        self.logger = logging.getLogger(self.__class__.__name__)

    def ingest_seo_report(self, report_text: str, language_label: str) -> None:
        """Store the SEO analysis report in the knowledge base for downstream use."""
        if not report_text:
            raise ValueError("SEO analysis report text is required for ingestion.")

        language_code = self.SUPPORTED_LANGUAGES.get(language_label, "en")
        annotated_text = f"[language={language_code}]\n{report_text}"
        self.logger.info("Ingesting SEO report into the knowledge base")
        self.rag_manager.add_to_knowledge_base(annotated_text)

    def extract_keywords(self, report_text: str, manual_keywords: Optional[List[str]], language_label: str) -> List[str]:
        """Extract SEO keywords and merge with any provided manual keywords."""
        keywords = [kw.strip() for kw in (manual_keywords or []) if kw.strip()]
        if not report_text:
            self.logger.warning("No SEO report provided; returning manual keywords only")
            return keywords

        prompt = (
            "You are an SEO specialist. Extract up to 12 high-impact keywords from the provided SEO analysis report. "
            "Return the keywords as a JSON array of strings and respond in the same language specified.\n\n"
            f"Language: {language_label}\n"
            f"SEO Analysis Report:\n{report_text}"
        )

        raw_keywords = ""
        try:
            self.logger.info("Requesting keyword extraction from OpenAI")
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            raw_keywords = response.choices[0].message.content.strip()
            extracted = json.loads(raw_keywords)
            if not isinstance(extracted, list):
                raise ValueError("Keyword response was not a list.")
            keywords.extend([kw.strip() for kw in extracted if isinstance(kw, str) and kw.strip()])
        except Exception as exc:
            self.logger.warning("Keyword extraction failed: %s", exc)
            try:
                # Fallback: attempt to split on commas
                fallback = [kw.strip() for kw in raw_keywords.split(",") if kw.strip()]
                keywords.extend(fallback)
            except Exception:
                self.logger.debug("Fallback keyword parsing failed.")

        # Remove duplicates while preserving order
        seen = set()
        unique_keywords: List[str] = []
        for kw in keywords:
            if kw.lower() not in seen:
                seen.add(kw.lower())
                unique_keywords.append(kw)

        self.logger.info("Compiled %d unique keywords", len(unique_keywords))
        return unique_keywords

    def _build_context(self, search_query: str) -> str:
        retriever = self.rag_manager.get_retriever()
        docs = retriever.invoke(search_query)
        return "\n\n".join(doc.page_content for doc in docs)

    def generate_article(self, topic: str, keywords: List[str], language_label: str) -> str:
        language_code = self.SUPPORTED_LANGUAGES.get(language_label, "en")
        search_query = topic if topic else ", ".join(keywords)
        context = self._build_context(search_query)
        keyword_block = ", ".join(keywords) if keywords else ""

        prompt = f"""
You are an experienced marketing copywriter creating an article in {language_label} (language code: {language_code}).
Incorporate the following SEO keywords naturally in the piece: {keyword_block if keyword_block else 'No specific keywords provided.'}

Reference insights from the internal knowledge base context provided below. If a section is not relevant, you may omit it but prioritize accuracy.

Knowledge Base Context:
{context if context else '[No additional context retrieved.]'}

Write a 800-1000 word article that is ready to publish immediately. Focus on actionable recommendations aligned with the SEO analysis. Do not include outlines or bullet-only drafts—deliver full prose content with headings.
"""

        self.logger.info("Generating long-form article")
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
        )
        return response.choices[0].message.content.strip()

    def generate_social_posts(self, article: str, keywords: List[str], language_label: str) -> dict:
        language_code = self.SUPPORTED_LANGUAGES.get(language_label, "en")
        prompt = f"""
Create compelling social media copy in {language_label} (language code: {language_code}) inspired by the following article.
Ensure the posts reflect the SEO focus using these keywords when natural: {', '.join(keywords) if keywords else 'None provided'}.

Article:
{article}

Provide a JSON object with keys 'facebook', 'instagram', and 'linkedin'. Each value should contain:
- A platform-tailored post in {language_label}
- Two relevant hashtags
- A suggested call-to-action
"""

        self.logger.info("Generating social posts for Facebook, Instagram, and LinkedIn")
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )

        raw_content = response.choices[0].message.content.strip()
        try:
            return json.loads(raw_content)
        except json.JSONDecodeError:
            self.logger.warning("Failed to parse JSON social posts; returning raw content.")
            return {"facebook": raw_content, "instagram": raw_content, "linkedin": raw_content}

    def run_pipeline(self, report_text: str, topic: str, manual_keywords: Optional[List[str]], language_label: str) -> dict:
        self.logger.info("Starting unified marketing pipeline")
        self.ingest_seo_report(report_text, language_label)
        keywords = self.extract_keywords(report_text, manual_keywords, language_label)
        article = self.generate_article(topic, keywords, language_label)
        social_posts = self.generate_social_posts(article, keywords, language_label)
        self.logger.info("Pipeline complete")
        return {
            "keywords": keywords,
            "article": article,
            "social_posts": social_posts,
        }

# GPT-4 Generation Node
def save_articles_data(articles):
    print(f"[DEBUG] Saving articles: {len(articles)} articles")
    try:
        # Get absolute path to ensure we're writing to the correct location
        file_path = os.path.abspath('articles_data.json')
        print(f"[DEBUG] Writing to file: {file_path}")
        
        # Ensure articles is a list
        if not isinstance(articles, list):
            print(f"[DEBUG] WARNING: articles is not a list, converting to list")
            if articles is None:
                articles = []
            else:
                articles = [articles]
        
        # Create a backup of the existing file if it exists
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            backup_path = f"{file_path}.bak"
            try:
                with open(file_path, 'r') as src, open(backup_path, 'w') as dst:
                    dst.write(src.read())
                print(f"[DEBUG] Created backup at {backup_path}")
            except Exception as e:
                print(f"[DEBUG] Error creating backup: {str(e)}")
        
        # Write the new data
        with open(file_path, 'w') as f:
            json.dump(articles, f, indent=2)
            f.flush()  # Force flush to disk
            os.fsync(f.fileno())  # Ensure data is written to disk
        
        # Verify the file was written correctly
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"[DEBUG] File saved successfully. Size: {file_size} bytes")
            
            # Read back the file to verify content
            with open(file_path, 'r') as f:
                content = f.read()
                print(f"[DEBUG] File content length: {len(content)} chars")
                
                # Verify JSON is valid
                try:
                    json.loads(content)
                    print(f"[DEBUG] JSON validation successful")
                except json.JSONDecodeError as e:
                    print(f"[DEBUG] ERROR: Invalid JSON written to file: {str(e)}")
                    # Restore from backup if available
                    backup_path = f"{file_path}.bak"
                    if os.path.exists(backup_path):
                        print(f"[DEBUG] Restoring from backup")
                        with open(backup_path, 'r') as src, open(file_path, 'w') as dst:
                            dst.write(src.read())
        else:
            print(f"[DEBUG] ERROR: File does not exist after saving!")
    except Exception as e:
        print(f"[DEBUG] ERROR saving articles: {str(e)}")
        import traceback
        print(traceback.format_exc())

def load_articles_data():
    try:
        file_path = os.path.abspath('articles_data.json')
        print(f"[DEBUG] Loading articles from: {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"[DEBUG] File does not exist, creating empty file")
            with open(file_path, 'w') as f:
                f.write('[]')
                f.flush()
                os.fsync(f.fileno())
            return []
        
        # Check file size
        file_size = os.path.getsize(file_path)
        print(f"[DEBUG] File exists, size: {file_size} bytes")
        
        if file_size == 0:
            print(f"[DEBUG] File is empty, initializing with empty array")
            with open(file_path, 'w') as f:
                f.write('[]')
                f.flush()
                os.fsync(f.fileno())
            return []
        
        # Read file content
        with open(file_path, 'r') as f:
            content = f.read()
            print(f"[DEBUG] Read file content: {content[:100]}{'...' if len(content) > 100 else ''}")
            
            if not content.strip():
                print(f"[DEBUG] File content is empty or whitespace, returning empty list")
                with open(file_path, 'w') as f:
                    f.write('[]')
                    f.flush()
                    os.fsync(f.fileno())
                return []
            
            # Try to parse JSON
            try:
                data = json.loads(content)
                
                # Validate that data is a list
                if not isinstance(data, list):
                    print(f"[DEBUG] WARNING: Loaded data is not a list, converting to list")
                    if data is None:
                        data = []
                    else:
                        data = [data]
                
                print(f"[DEBUG] Successfully parsed JSON, found {len(data)} articles")
                return data
            except json.JSONDecodeError as e:
                print(f"[DEBUG] JSON parsing error: {str(e)}")
                print(f"[DEBUG] Content causing error: {content}")
                
                # Check for backup file
                backup_path = f"{file_path}.bak"
                if os.path.exists(backup_path):
                    print(f"[DEBUG] Attempting to restore from backup")
                    try:
                        with open(backup_path, 'r') as f:
                            backup_content = f.read()
                            backup_data = json.loads(backup_content)
                            # Write the backup data back to the main file
                            with open(file_path, 'w') as f:
                                json.dump(backup_data, f, indent=2)
                                f.flush()
                                os.fsync(f.fileno())
                            print(f"[DEBUG] Successfully restored from backup")
                            return backup_data
                    except Exception as backup_error:
                        print(f"[DEBUG] Error restoring from backup: {str(backup_error)}")
                
                # If file is corrupted and no backup, reset it
                print(f"[DEBUG] Resetting corrupted file to empty array")
                with open(file_path, 'w') as f:
                    f.write('[]')
                    f.flush()
                    os.fsync(f.fileno())
                return []
    except Exception as e:
        print(f"[DEBUG] Error loading articles: {str(e)}")
        import traceback
        print(traceback.format_exc())
        
        # Ensure the file exists with valid JSON
        try:
            with open(file_path, 'w') as f:
                f.write('[]')
                f.flush()
                os.fsync(f.fileno())
            print(f"[DEBUG] Reset file after error")
        except Exception as reset_error:
            print(f"[DEBUG] Error resetting file: {str(reset_error)}")
        
        return []

def generate_article(topic, publish_date, include_feedback=False, target_audience=None, tone=None, word_count=None):
    print(f"[DEBUG] generate_article called for topic: {topic}")
    # Load any existing feedback for this topic
    feedback = ""
    if include_feedback:
        articles = load_articles_data()
        for article in articles:
            # Check if article has 'topic' key before comparing
            if isinstance(article, dict) and 'topic' in article and article['topic'] == topic:
                structured_feedback = format_feedback_for_prompt(article.get('feedback', {}))
                quick_feedback = article.get('quick_feedback', '')
                
                if structured_feedback or quick_feedback:
                    feedback = "\n\nPrevious feedback to incorporate:"
                    if structured_feedback:
                        feedback += f"\n{structured_feedback}"
                    if quick_feedback:
                        feedback += f"\nQuick Feedback: {quick_feedback}"
    
    # Build the prompt with additional parameters
    prompt = f"""Write a complete, publication-ready article about {topic}. Do not include any planning notes or outlines - write the actual article content that will be published on {publish_date}.

The article should be written in a way that's ready to be published immediately without further editing. Do not include any meta-information or planning details - only write the actual article content that readers will see."""
    
    if target_audience:
        prompt += f"\nTarget Audience: {target_audience}"
    if tone:
        prompt += f"\nTone: {tone}"
    if word_count:
        prompt += f"\nTarget Word Count: {word_count} words"
    if feedback:
        prompt += feedback
        
    print(f"[DEBUG] Built prompt for article generation, length: {len(prompt)}")
    print(f"[DEBUG] Prompt preview: {prompt[:200]}...")
    
    # Add content guidelines
    prompt += """

Article Structure:
1. Start with a clear, engaging headline at the top
2. Write a compelling introduction that hooks the reader
3. Organize the main content with descriptive subheadings
4. Include practical examples and real-world applications
5. End with a strong conclusion and clear call-to-action

Formatting Requirements:
- Write in a clear, direct style
- Use short paragraphs (2-3 sentences each)
- Include subheadings to break up the text
- Use bullet points or numbered lists where appropriate
- Incorporate relevant statistics or data points
- End with a clear takeaway message

Remember: Write the actual article content, not a plan or outline. The output should be ready to publish as-is.
    """
    
    try:
        print(f"[DEBUG] Making OpenAI API call with model: gpt-3.5-turbo")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Using gpt-3.5-turbo instead of gpt-4 for faster response
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        content = response.choices[0].message.content
        print(f"[DEBUG] OpenAI API call successful, response length: {len(content)}")
        print(f"[DEBUG] Response preview: {content[:100]}...")
    except Exception as e:
        print(f"[DEBUG] OpenAI API call failed: {str(e)}")
        import traceback
        print(f"[DEBUG] Error traceback: {traceback.format_exc()}")
        content = f"[ARTICLE GENERATION FAILED]\nError: {str(e)}\nPrompt: {prompt}"
    
    # Save the generated article
    articles = load_articles_data()
    print(f"[DEBUG] Loaded articles before append: {len(articles)} articles")
    
    new_article = {
        "topic": topic,
        "publish_date": publish_date,
        "content": content,
        "status": "generated",
        "feedback": "",
        "generated_at": datetime.now().isoformat(),
        "revisions": []
    }
    
    print(f"[DEBUG] New article object created: {new_article['topic']}")
    articles.append(new_article)
    print(f"[DEBUG] Articles after append: {len(articles)} articles")
    
    try:
        save_articles_data(articles)
        print(f"[DEBUG] Articles saved successfully")
    except Exception as save_error:
        print(f"[DEBUG] Error saving articles: {str(save_error)}")
        import traceback
        print(f"[DEBUG] Save error traceback: {traceback.format_exc()}")
    
    return content

def format_feedback_for_prompt(feedback_data):
    prompt_parts = []
    
    if isinstance(feedback_data, dict):
        for category, data in feedback_data.items():
            if isinstance(data, dict):
                category_name = category.replace('_', ' ').title()
                prompt_parts.append(f"\n{category_name} Feedback:")
                
                # Add ratings
                ratings = [f"{k.replace('_', ' ').title()}: {v}/5" 
                          for k, v in data.items() if k != 'feedback' and v is not None]
                if ratings:
                    prompt_parts.append("Ratings: " + ", ".join(ratings))
                
                # Add textual feedback
                if data.get('feedback'):
                    prompt_parts.append(f"Suggestions: {data['feedback']}")
    else:
        # Handle legacy or quick feedback format
        prompt_parts.append(f"\nGeneral Feedback: {feedback_data}")
    
    return "\n".join(prompt_parts)

def revise_article(article_idx):
    articles = load_articles_data()
    
    # Check if article_idx is an integer or a dictionary
    if isinstance(article_idx, int):
        # Find the article by topic and publish_date instead of by index
        # First, get the article from the filtered list to know what we're looking for
        filtered_articles = [a for a in articles if isinstance(a, dict) and 'status' in a and 'topic' in a and 'publish_date' in a]
        if 'status_filter' in globals() and status_filter != "All":
            filtered_articles = [a for a in filtered_articles if a['status'] == status_filter]
        
        if article_idx >= len(filtered_articles):
            print(f"[DEBUG] Invalid article index: {article_idx}, max: {len(filtered_articles)-1}")
            return
        
        target_article = filtered_articles[article_idx]
        
        # Find the same article in the original list
        article = None
        for i, a in enumerate(articles):
            if isinstance(a, dict) and 'topic' in a and 'publish_date' in a:
                if a['topic'] == target_article['topic'] and a['publish_date'] == target_article['publish_date']:
                    article = a
                    break
    else:
        # article_idx is actually the article itself
        target_article = article_idx
        
        # Find the same article in the original list
        article = None
        for i, a in enumerate(articles):
            if isinstance(a, dict) and 'topic' in a and 'publish_date' in a:
                if a['topic'] == target_article['topic'] and a['publish_date'] == target_article['publish_date']:
                    article = a
                    break
    
    if not article:
        print(f"[DEBUG] Could not find article in original list")
        return
    
    # Store the current version in revisions
    article['revisions'].append({
        'content': article['content'],
        'feedback': article.get('feedback', ''),
        'quick_feedback': article.get('quick_feedback', ''),
        'date': datetime.now().isoformat()
    })
    
    # Format feedback for the prompt
    structured_feedback = format_feedback_for_prompt(article.get('feedback', {}))
    quick_feedback = article.get('quick_feedback', '')
    
    # Combine both types of feedback
    combined_feedback = structured_feedback
    if quick_feedback:
        combined_feedback += f"\n\nQuick Feedback: {quick_feedback}"
    
    # Generate new version incorporating feedback
    new_content = generate_article(
        article['topic'],
        article['publish_date'],
        include_feedback=True
    )
    
    # Update article
    article['content'] = new_content
    article['feedback'] = ""
    article['status'] = "revised"
    save_articles_data(articles)

def publish_to_wordpress(article, wp_url, wp_username, wp_app_password):
    """Publish an article to WordPress using the WordPress REST API"""
    try:
        # Create credentials token
        credentials = f"{wp_username}:{wp_app_password}"
        token = base64.b64encode(credentials.encode()).decode('utf-8')
        headers = {'Authorization': f'Basic {token}'}
        
        # Prepare the post data
        post_data = {
            'title': article['topic'],
            'content': article['content'],
            'status': 'publish',
            'date': datetime.now().isoformat()
        }
        
        # Add tags if available
        if 'tags' in article:
            post_data['tags'] = article['tags']
            
        # Make the API request
        api_url = f"{wp_url}/wp-json/wp/v2/posts"
        response = requests.post(api_url, headers=headers, json=post_data)
        
        if response.status_code in [200, 201]:
            # Update article with WordPress post ID and URL
            article['wp_post_id'] = response.json().get('id')
            article['wp_post_url'] = response.json().get('link')
            article['status'] = 'published'
            article['published_at'] = datetime.now().isoformat()
            article['published_platform'] = 'wordpress'
            return True, article['wp_post_url']
        else:
            return False, f"Error: {response.status_code} - {response.text}"
    
    except Exception as e:
        return False, f"Error: {str(e)}"

def publish_to_medium(article, medium_token, medium_tags=None):
    """Publish an article to Medium using the Medium API"""
    try:
        # Set up headers with the integration token
        headers = {
            'Authorization': f'Bearer {medium_token}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        # First, get the user ID
        user_url = 'https://api.medium.com/v1/me'
        user_response = requests.get(url=user_url, headers=headers)
        
        if user_response.status_code != 200:
            return False, f"Error getting Medium user ID: {user_response.status_code} - {user_response.text}"
        
        user_id = user_response.json()['data']['id']
        
        # Prepare the post data
        post_data = {
            "title": article['topic'],
            "contentFormat": "markdown",
            "content": article['content'],
            "publishStatus": "public"
        }
        
        # Add tags if available
        if medium_tags:
            post_data["tags"] = medium_tags
        elif 'tags' in article:
            post_data["tags"] = article['tags']
        
        # Make the API request to publish
        post_url = f'https://api.medium.com/v1/users/{user_id}/posts'
        post_response = requests.post(url=post_url, headers=headers, data=json.dumps(post_data))
        
        if post_response.status_code in [200, 201]:
            # Update article with Medium post ID and URL
            article['medium_post_id'] = post_response.json()['data'].get('id')
            article['medium_post_url'] = post_response.json()['data'].get('url')
            article['status'] = 'published'
            article['published_at'] = datetime.now().isoformat()
            article['published_platform'] = 'medium'
            return True, article['medium_post_url']
        else:
            return False, f"Error: {post_response.status_code} - {post_response.text}"
    
    except Exception as e:
        return False, f"Error: {str(e)}"

def publish_article(article, platform, credentials):
    """Publish an article to the selected platform"""
    if platform == 'wordpress':
        return publish_to_wordpress(
            article, 
            credentials['wp_url'], 
            credentials['wp_username'], 
            credentials['wp_app_password']
        )
    elif platform == 'medium':
        return publish_to_medium(
            article, 
            credentials['medium_token'],
            credentials.get('medium_tags')
        )
    else:
        return False, "Unsupported platform"

def simulate_publish(article):
    # Keep the original simulation functionality for backward compatibility
    article['status'] = 'published'
    article['published_at'] = datetime.now().isoformat()
    article['published_platform'] = 'simulated'
    return True, "Simulated publication"

def check_and_publish_scheduled():
    articles = load_articles_data()
    current_date = datetime.now().date()
    
    for article in articles:
        publish_date = datetime.strptime(article['publish_date'], '%Y-%m-%d').date()
        if publish_date <= current_date and article['status'] == 'generated':
            if simulate_publish(article):
                article['status'] = 'published'
                article['published_at'] = datetime.now().isoformat()
    
    save_articles_data(articles)

def schedule_article(topic, publish_date, target_audience=None, tone=None, word_count=None):
    print(f"[DEBUG] Starting schedule_article for topic: {topic}, date: {publish_date}")
    
    # First, check if articles_data.json exists and is valid
    try:
        file_path = os.path.abspath('articles_data.json')
        if not os.path.exists(file_path):
            print(f"[DEBUG] articles_data.json does not exist, creating it")
            with open(file_path, 'w') as f:
                f.write('[]')
                f.flush()
                os.fsync(f.fileno())
    except Exception as e:
        print(f"[DEBUG] Error checking/creating articles_data.json: {str(e)}")
    
    # Generate the article immediately
    try:
        print(f"[DEBUG] Calling generate_article for topic: {topic}")
        print(f"[DEBUG] OpenAI client initialized: {client is not None}")
        
        # Test OpenAI API connection
        try:
            test_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                temperature=0.7
            )
            print(f"[DEBUG] OpenAI API test successful: {test_response.choices[0].message.content}")
        except Exception as api_test_error:
            print(f"[DEBUG] OpenAI API test failed: {str(api_test_error)}")
        
        content = generate_article(
            topic,
            publish_date,
            target_audience=target_audience,
            tone=tone,
            word_count=word_count
        )
        print(f"[DEBUG] Article content generated, length: {len(content)}")
    except Exception as e:
        print(f"[DEBUG] Error generating article content: {str(e)}")
        import traceback
        print(f"[DEBUG] Error traceback: {traceback.format_exc()}")
        content = f"[ARTICLE GENERATION FAILED] Error: {str(e)}"
    
    # Schedule the publishing job
    try:
        publish_date_obj = datetime.strptime(publish_date, '%Y-%m-%d') if isinstance(publish_date, str) else publish_date
        scheduler.add_job(
            check_and_publish_scheduled,
            trigger=DateTrigger(run_date=publish_date_obj),
            id=f"publish_{topic}_{publish_date}"
        )
        print(f"[DEBUG] Publishing job scheduled for {publish_date}")
    except Exception as e:
        print(f"[DEBUG] Error scheduling job: {str(e)}")
    
    # Create the new article object
    new_article = {
        "topic": topic,
        "publish_date": str(publish_date),
        "target_audience": target_audience,
        "tone": tone,
        "word_count": word_count,
        "status": "scheduled",
        "content": content,
        "feedback": "",
        "generated_at": datetime.now().isoformat(),
        "revisions": []
    }
    
    # Save the scheduled article to persistent storage
    try:
        # Load existing articles
        articles = load_articles_data()
        print(f"[DEBUG] Loaded articles for scheduling: {len(articles)} articles")
        
        # Add the new article
        articles.append(new_article)
        print(f"[DEBUG] Added new article to articles list, now has {len(articles)} articles")
        
        # Save to file
        save_articles_data(articles)
        print(f"[DEBUG] Saved articles data to file")
        
        # Update session state to match the persistent storage
        st.session_state.scheduled_articles = articles
        print(f"[DEBUG] Updated session state, now has {len(st.session_state.scheduled_articles)} scheduled articles")
    except Exception as e:
        print(f"[DEBUG] Error in scheduling article: {str(e)}")
        import traceback
        print(traceback.format_exc())
        st.error(f"Error scheduling article: {str(e)}")
    
    return content

def generate_content_with_rag(user_goal, headlines, retriever):
    rag_context = retriever.invoke(user_goal)
    rag_text = "\n".join([doc.page_content for doc in rag_context])

    # Extract just the titles from the headlines tuples
    headline_titles = [h[0] for h in headlines]

    prompt = f"""You are a content strategist.

User Goal: {user_goal}
Trending Headlines:
{chr(10).join(f"- {h}" for h in headline_titles)}

Internal Guidelines:
{rag_text}

Generate:
1. A 30-day content calendar with 5 detailed article topics (include topic, format, platform, and target audience)
2. For each topic, provide a brief description of the angle and key points to cover
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    content_plan = response.choices[0].message.content
    
    # Parse the content plan and schedule articles
    try:
        st.subheader("📅 Content Calendar and Article Generation")
        st.write(content_plan)
        
        # Create a 6-month date range
        start_date = datetime.now()
        end_date = start_date + timedelta(days=180)
        
        # Article configuration
        st.subheader("Configure and Schedule Article")
        selected_topic = st.text_input("Enter the topic for the article")
        
        target_audience = st.selectbox(
            "Target Audience",
            ["General", "Technical", "Business", "Marketing", "Developers", "Executives"],
            key="content_plan_audience_main"  # Changed key to be unique
        )
        
        tone = st.selectbox(
            "Content Tone",
            ["Professional", "Casual", "Technical", "Conversational", "Formal", "Inspirational"],
            key="content_plan_tone_main"  # Changed key to be unique
        )
        
        word_count = st.number_input(
            "Target Word Count",
            min_value=500,
            max_value=5000,
            value=1000,
            step=100,
            key="content_plan_word_count_main"  # <-- Make this unique
        )
        
        selected_date = st.date_input(
            "Select publication date",
            min_value=start_date.date(),
            max_value=end_date.date()
        )
        
        if st.button("Generate and Schedule Article"):
            if selected_topic:
                schedule_article(
                    selected_topic,
                    selected_date,
                    target_audience=target_audience,
                    tone=tone,
                    word_count=word_count
                )
                st.success(f"Article generated and scheduled for {selected_date}")
            else:
                st.error("Please enter a topic for the article")
    
    except Exception as e:
        st.error(f"Error scheduling article: {str(e)}")
    
    return content_plan

# Streamlit UI with enhanced features
st.title("🧠 AI Content Planner")

# Sidebar for filters and knowledge base management
with st.sidebar:
    st.header("Settings & Knowledge Base")
    
    # News API Filters
    language = st.selectbox("Select Language", ["en", "es", "fr", "de"], key="sidebar_language")
    sort_by = st.selectbox("Sort News By", ["relevancy", "popularity", "publishedAt"], key="sidebar_sort_by")
    from_date = st.date_input("From Date", key="sidebar_from_date")
    
    # Knowledge Base Management
    st.subheader("Knowledge Base Management")
    new_guidelines = st.text_area("Add New Guidelines")
    if st.button("Add to Knowledge Base"):
        rag_manager = RAGManager()
        rag_manager.add_to_knowledge_base(new_guidelines)
        st.success("Guidelines added successfully!")

# Main content area
user_goal = st.text_input("Enter your marketing goal:", "Promote AI tools for startups")
keywords_input = st.text_input("Enter keywords (comma-separated):", "AI productivity, startup automation")

# Add this near the top of the file where other session state variables are initialized
if 'show_article_config' not in st.session_state:
    st.session_state.show_article_config = False
if 'content_plan_result' not in st.session_state:
    st.session_state.content_plan_result = None
if 'headlines_result' not in st.session_state:
    st.session_state.headlines_result = []
if 'pipeline_result' not in st.session_state:
    st.session_state.pipeline_result = None

st.subheader("🔁 Unified SEO-to-Content Pipeline")
pipeline_language = st.selectbox(
    "Select pipeline language",
    list(MarketingPipeline.SUPPORTED_LANGUAGES.keys()),
    key="pipeline_language",
)

pipeline_topic = st.text_input(
    "Primary article topic or focus",
    value="",
    key="pipeline_topic",
)

seo_report_text = st.text_area(
    "Paste SEO analysis report",
    value="",
    height=200,
    key="pipeline_report_text",
)

seo_report_file = st.file_uploader(
    "Upload SEO analysis report (optional)",
    type=["txt", "md"],
    key="pipeline_report_file",
)

manual_keywords_input = st.text_input(
    "Optional manual keywords (comma separated)",
    value="",
    key="pipeline_manual_keywords",
)

if st.button("Run Unified Pipeline", key="pipeline_run_button"):
    combined_report = seo_report_text.strip()
    if seo_report_file is not None:
        try:
            uploaded_text = seo_report_file.read().decode("utf-8")
            combined_report = f"{combined_report}\n\n{uploaded_text}" if combined_report else uploaded_text
        except Exception as upload_error:
            st.error(f"Unable to read uploaded report: {upload_error}")
            combined_report = combined_report or ""

    manual_keywords = [kw.strip() for kw in manual_keywords_input.split(",") if kw.strip()]

    if not combined_report:
        st.error("Please provide an SEO analysis report by pasting text or uploading a file.")
        st.session_state.pipeline_result = None
    else:
        try:
            pipeline_runner = MarketingPipeline(client, RAGManager())
            pipeline_output = pipeline_runner.run_pipeline(
                report_text=combined_report,
                topic=pipeline_topic,
                manual_keywords=manual_keywords,
                language_label=pipeline_language,
            )
            st.session_state.pipeline_result = pipeline_output
            st.success("Pipeline completed successfully. Review the generated assets below.")
        except Exception as pipeline_error:
            st.session_state.pipeline_result = None
            st.error(f"Pipeline execution failed: {pipeline_error}")

if st.session_state.pipeline_result:
    pipeline_output = st.session_state.pipeline_result
    st.markdown("### Extracted Keywords")
    if pipeline_output["keywords"]:
        st.write(", ".join(pipeline_output["keywords"]))
    else:
        st.info("No keywords were extracted or provided.")

    st.markdown("### Generated Article")
    st.text_area(
        "Article",
        value=pipeline_output["article"],
        height=400,
        key="pipeline_article_output",
        disabled=True,
    )

    social_posts = pipeline_output.get("social_posts", {})
    st.markdown("### Social Media Posts")
    for platform in ["facebook", "instagram", "linkedin"]:
        content = social_posts.get(platform)
        if content:
            st.markdown(f"**{platform.title()}**")
            if isinstance(content, str):
                st.write(content)
            else:
                st.json(content)
        else:
            st.write(f"No {platform} post generated.")

if st.button("Generate Plan"):
    keywords = [k.strip() for k in keywords_input.split(",")]
    mcp = MCPGoogleNewsConnector(newsapi)
    headlines = mcp.fetch_headlines(
        keywords,
        language=language,
        sort_by=sort_by,
        from_date=from_date.strftime('%Y-%m-%d') if from_date else None
    )
    
    rag_manager = RAGManager()
    retriever = rag_manager.get_retriever()
    result = generate_content_with_rag(user_goal, headlines, retriever)
    
    # Store results in session state
    st.session_state.content_plan_result = result
    st.session_state.headlines_result = headlines
    st.session_state.show_article_config = True

# Display results if they exist in session state
if st.session_state.show_article_config:
    # Display headlines with metadata
    st.subheader("📌 Headlines")
    for title, source, date in st.session_state.headlines_result:
        st.markdown(f"- **{title}** | {source} | {date}")

    st.subheader("📅 Content Plan")
    st.text_area("Generated Plan", st.session_state.content_plan_result, height=400)
    
    # Article configuration
    st.subheader("Configure and Schedule Article")
    selected_topic = st.text_input("Enter the topic for the article", key="article_topic")
    
    target_audience = st.selectbox(
        "Target Audience",
        ["General", "Technical", "Business", "Marketing", "Developers", "Executives"],
        key="content_plan_audience"
    )
    
    tone = st.selectbox(
        "Content Tone",
        ["Professional", "Casual", "Technical", "Conversational", "Formal", "Inspirational"],
        key="content_plan_tone"
    )
    
    word_count = st.number_input(
        "Target Word Count",
        min_value=500,
        max_value=5000,
        value=1000,
        step=100,
        key="content_plan_word_count_config"  # <-- Make this unique
    )
    
    # Add a key to the date_input to maintain its state
    start_date = datetime.now()
    end_date = start_date + timedelta(days=180)
    selected_date = st.date_input(
        "Select publication date",
        min_value=start_date.date(),
        max_value=end_date.date(),
        key="publication_date"
    )
    
    if st.button("Generate and Schedule Article", key="generate_article_btn"):
        if selected_topic:
            schedule_article(
                selected_topic,
                selected_date,
                target_audience=target_audience,
                tone=tone,
                word_count=word_count
            )
            st.success(f"Article generated and scheduled for {selected_date}")
        else:
            st.error("Please enter a topic for the article")

    # Enhanced feedback system
    col1, col2 = st.columns(2)
    with col1:
        feedback = st.slider("Rate the usefulness of this plan (1-5):", 1, 5, 3, key="feedback_slider")
    with col2:
        feedback_text = st.text_input("Additional feedback (optional):", key="feedback_text")

    if st.button("Submit Feedback"):
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "rating": feedback,
            "comment": feedback_text,
            "goal": user_goal
        }
        st.session_state.feedback_data.append(feedback_entry)
        
        # Save feedback to file
        with open('feedback_data.json', 'w') as f:
            json.dump(st.session_state.feedback_data, f)
        
        st.success(f"Thanks! You rated this {feedback}/5. Feedback saved.")

# Display content calendar and management
st.subheader("📋 Content Management")
tab1, tab2, tab3 = st.tabs(["Publication Calendar", "Scheduled Articles", "Content Manager"])

with tab1:
    st.subheader("📅 Upcoming Publications")
    
    # Create a calendar view of scheduled content
    articles = load_articles_data()
    if articles:
        # Group articles by publish date
        publication_calendar = {}
        for article in articles:
            # Check if article has all required keys
            if not isinstance(article, dict) or 'status' not in article or 'publish_date' not in article:
                print(f"[DEBUG] Skipping invalid article: {article}")
                continue
                
            if article['status'] != 'published':
                date = article['publish_date']
                if date not in publication_calendar:
                    publication_calendar[date] = []
                publication_calendar[date].append(article)
        
        # Display calendar
        if publication_calendar:
            for date in sorted(publication_calendar.keys()):
                st.markdown(f"### {date}")
                for i, article in enumerate(publication_calendar[date]):
                    status_emoji = '🟡' if article['status'] == 'generated' else '🔵'
                    st.markdown(f"{status_emoji} **{article['topic']}**")
                    with st.expander("Preview Content"):
                        st.text_area(
                            "Content Preview",
                            article['content'][:500] + "...",
                            height=100,
                            key=f"preview_{date}_{i}_{article['topic']}"
                        )
        else:
            st.info("No upcoming publications scheduled.")
    else:
        st.info("No content in the system.")

with tab2:
    st.subheader("Schedule New Article")
    
    # Article configuration
    target_audience = st.selectbox(
        "Target Audience",
        ["General", "Technical", "Business", "Marketing", "Developers", "Executives"],
        key="schedule_audience"
    )
    
    tone = st.selectbox(
        "Content Tone",
        ["Professional", "Casual", "Technical", "Conversational", "Formal", "Inspirational"],
        key="schedule_tone"
    )
    
    word_count = st.number_input(
        "Target Word Count",
        min_value=500,
        max_value=5000,
        value=1000,
        step=100,
        key="schedule_word_count"  
    )
    
    # Display scheduled articles
    st.subheader("Scheduled Articles")
    if st.session_state.scheduled_articles:
        scheduled_df = pd.DataFrame(st.session_state.scheduled_articles)
        st.dataframe(scheduled_df)
    else:
        st.info("No scheduled articles yet.")

def get_article_review(idx):
    review_data = {}
    
    # Content Quality
    st.subheader("📝 Content Quality")
    review_data['content_quality'] = {
        'rating': st.slider("Overall content quality", 1, 5, 3, key=f'content_quality_{idx}'),
        'clarity': st.slider("Clarity and readability", 1, 5, 3, key=f'clarity_{idx}'),
        'accuracy': st.slider("Information accuracy", 1, 5, 3, key=f'accuracy_{idx}'),
        'feedback': st.text_area("Content improvement suggestions", key=f'content_feedback_{idx}')
    }
    
    # SEO Assessment
    st.subheader("🔍 SEO Assessment")
    review_data['seo'] = {
        'keywords': st.slider("Keyword usage and placement", 1, 5, 3, key=f'keywords_{idx}'),
        'meta_elements': st.slider("Meta elements (title, description)", 1, 5, 3, key=f'meta_{idx}'),
        'structure': st.slider("Content structure and headings", 1, 5, 3, key=f'structure_{idx}'),
        'feedback': st.text_area("SEO improvement suggestions", key=f'seo_feedback_{idx}')
    }
    
    # Target Audience Alignment
    st.subheader("👥 Audience Alignment")
    review_data['audience'] = {
        'relevance': st.slider("Content relevance to target audience", 1, 5, 3, key=f'relevance_{idx}'),
        'engagement': st.slider("Engagement potential", 1, 5, 3, key=f'engagement_{idx}'),
        'value': st.slider("Value proposition clarity", 1, 5, 3, key=f'value_{idx}'),
        'feedback': st.text_area("Audience alignment suggestions", key=f'audience_feedback_{idx}')
    }
    
    # Editorial Guidelines
    st.subheader("✍️ Editorial Review")
    review_data['editorial'] = {
        'tone': st.slider("Tone and style consistency", 1, 5, 3, key=f'tone_{idx}'),
        'grammar': st.slider("Grammar and spelling", 1, 5, 3, key=f'grammar_{idx}'),
        'formatting': st.slider("Formatting and presentation", 1, 5, 3, key=f'formatting_{idx}'),
        'feedback': st.text_area("Editorial suggestions", key=f'editorial_feedback_{idx}')
    }
    
    return review_data

with tab3:
    articles = load_articles_data()
    if articles:
        # Filter options
        status_filter = st.selectbox(
            "Filter by Status",
            ["All", "generated", "revised", "published"]
        )
        
        # Filter out invalid articles and apply status filter
        valid_articles = [a for a in articles if isinstance(a, dict) and 'status' in a and 'topic' in a and 'publish_date' in a]
        print(f"[DEBUG] Found {len(valid_articles)} valid articles out of {len(articles)} total")
        
        filtered_articles = valid_articles
        if status_filter != "All":
            filtered_articles = [a for a in valid_articles if a['status'] == status_filter]
        
        for idx, article in enumerate(filtered_articles):
            status_color = {
                'generated': '🟡',
                'revised': '🔵',
                'published': '🟢'
            }.get(article['status'], '⚪')
            
            with st.expander(f"{status_color} {article['topic']} - {article['publish_date']}"):
                st.text_area("Content", article['content'], height=200, key=f"content_{idx}")
                
                # Review and Feedback Section
                if article['status'] != 'published':
                    st.subheader("📋 Review and Feedback")
                    review_tab1, review_tab2 = st.tabs(["Structured Review", "Quick Feedback"])
                    
                    with review_tab1:
                        # Fix: Pass the idx parameter to get_article_review
                        review_data = get_article_review(idx)
                        if st.button("Submit Review", key=f"submit_review_{idx}"):
                            article['feedback'] = review_data
                            save_articles_data(articles)
                            st.success("Review submitted successfully!")
                    
                    with review_tab2:
                        quick_feedback = st.text_area(
                            "Quick Feedback",
                            article.get('quick_feedback', ''),
                            key=f"quick_feedback_{idx}"
                        )
                        if quick_feedback != article.get('quick_feedback', ''):
                            article['quick_feedback'] = quick_feedback
                            save_articles_data(articles)
                
                # Action Buttons
                col1, col2 = st.columns(2)
                with col1:
                    if (article.get('feedback') or article.get('quick_feedback')) and article['status'] != 'published':
                        if st.button("Generate Revision", key=f"revise_{idx}"):
                            # Find the index of this article in the original list
                            original_idx = None
                            all_articles = load_articles_data()
                            for i, a in enumerate(all_articles):
                                if isinstance(a, dict) and 'topic' in a and 'publish_date' in a:
                                    if a['topic'] == article['topic'] and a['publish_date'] == article['publish_date']:
                                        original_idx = i
                                        break
                            
                            if original_idx is not None:
                                revise_article(original_idx)
                                st.rerun()
                            else:
                                st.error("Could not find article in the database")
                
                with col2:
                    if article['status'] == 'generated' or article['status'] == 'revised':
                        # Create a dropdown for platform selection
                        platform = st.selectbox(
                            "Select Platform",
                            ["Simulate", "WordPress", "Medium"],
                            key=f"platform_{idx}"
                        )
                        
                        # Show credential inputs based on selected platform
                        if platform == "WordPress":
                            wp_url = st.text_input("WordPress URL (e.g., https://example.com)", key=f"wp_url_{idx}")
                            wp_username = st.text_input("WordPress Username", key=f"wp_username_{idx}")
                            wp_app_password = st.text_input("WordPress App Password", type="password", key=f"wp_password_{idx}")
                            credentials = {
                                'wp_url': wp_url,
                                'wp_username': wp_username,
                                'wp_app_password': wp_app_password
                            }
                        elif platform == "Medium":
                            medium_token = st.text_input("Medium Integration Token", type="password", key=f"medium_token_{idx}")
                            medium_tags = st.text_input("Tags (comma-separated)", key=f"medium_tags_{idx}")
                            credentials = {
                                'medium_token': medium_token,
                                'medium_tags': [tag.strip() for tag in medium_tags.split(",")] if medium_tags else []
                            }
                        else:  # Simulate
                            credentials = {}
                        
                        if st.button("Publish Now", key=f"publish_{idx}"):
                            if platform == "Simulate":
                                success = simulate_publish(article)
                                if success:
                                    save_articles_data(articles)
                                    st.success("Article simulated as published")
                                    st.rerun()
                            else:
                                # Validate credentials
                                if platform == "WordPress" and (not wp_url or not wp_username or not wp_app_password):
                                    st.error("Please provide all WordPress credentials")
                                elif platform == "Medium" and not medium_token:
                                    st.error("Please provide Medium integration token")
                                else:
                                    with st.spinner(f"Publishing to {platform}..."):
                                        success, message = publish_article(article, platform.lower(), credentials)
                                        if success:
                                            save_articles_data(articles)
                                            st.success(f"Article published successfully to {platform}!")
                                            st.markdown(f"[View Published Article]({message})")
                                            st.rerun()
                                        else:
                                            st.error(f"Failed to publish: {message}")
                
                # Show revision history
                if article.get('revisions'):
                    with st.expander("📚 Revision History"):
                        for rev_idx, revision in enumerate(article['revisions']):
                            st.markdown(f"**Revision {rev_idx + 1}** - {revision['date']}")
                            st.text_area(
                                "Content",
                                revision['content'],
                                height=100,
                                key=f"rev_{idx}_{rev_idx}"
                            )
                            if revision.get('feedback'):
                                with st.expander("Review Details"):
                                    for category, data in revision['feedback'].items():
                                        st.markdown(f"**{category.replace('_', ' ').title()}**")
                                        if isinstance(data, dict):
                                            for metric, value in data.items():
                                                if metric != 'feedback':
                                                    st.write(f"{metric.replace('_', ' ').title()}: {value}/5")
                                            if data.get('feedback'):
                                                st.write(f"Feedback: {data['feedback']}")
                
                status_text = f"Status: {article['status']} | Generated: {article['generated_at']}"
                if article.get('published_at'):
                    status_text += f" | Published: {article['published_at']}"
                    if article.get('published_platform'):
                        status_text += f" | Platform: {article.get('published_platform')}"
                    if article.get('wp_post_url'):
                        status_text += f" | [View on WordPress]({article.get('wp_post_url')})"
                    if article.get('medium_post_url'):
                        status_text += f" | [View on Medium]({article.get('medium_post_url')})"
                st.info(status_text)
    else:
        st.info("No generated articles yet.")

# Display feedback history
if st.session_state.feedback_data:
    st.subheader("📊 Feedback History")
    df = pd.DataFrame(st.session_state.feedback_data)
    st.dataframe(df)
