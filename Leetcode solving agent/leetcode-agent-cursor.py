


import asyncio
import json
import re
from typing import Dict, List, Optional, TypedDict, Annotated
from dataclasses import dataclass
from playwright.async_api import async_playwright, Page, Browser
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
import time
import sys

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


# For Jupyter compatibility - handle event loop
try:
    import nest_asyncio
    nest_asyncio.apply()
    print("✅ Nested event loop enabled for Jupyter")
except ImportError:
    print("⚠️  nest_asyncio not installed. Run: pip install nest_asyncio")

# State definition for LangGraph with better type annotations
class AgentState(TypedDict):
    problem_url: Annotated[str, "URL of the LeetCode problem"]
    problem_title: Annotated[str, "Title of the problem"]
    problem_description: Annotated[str, "Problem description text"]
    problem_difficulty: Annotated[str, "Problem difficulty level"]
    solution_code: Annotated[str, "Generated solution code"]
    test_results: Annotated[List[Dict], "Results from test case execution"]
    submission_result: Annotated[Dict, "Result from problem submission"]
    runtime_stats: Annotated[Dict, "Runtime and memory statistics"]
    optimization_suggestions: Annotated[str, "AI-generated optimization suggestions"]
    current_step: Annotated[str, "Current step in the workflow"]
    error_message: Annotated[str, "Error message if any step fails"]
    attempt_count: Annotated[int, "Number of solution attempts made"]

@dataclass
class LeetCodeConfig:
    username: str
    password: str
    headless: bool = True
    timeout: int = 60000
    max_attempts: int = 3

class LeetCodeAutomation:
    def __init__(self, config: LeetCodeConfig):
        self.config = config
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.playwright = None
        
    async def __aenter__(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=self.config.headless,
            args=['--no-sandbox', '--disable-blink-features=AutomationControlled']
        )
        self.page = await self.browser.new_page()
        
        # Set user agent to avoid detection
        await self.page.set_extra_http_headers({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        
        await self.page.set_viewport_size({"width": 1920, "height": 1080})
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.page:
            await self.page.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
    
    async def debug_page_state(self):
        """Debug helper to understand page state"""
        if not self.page:
            print("⚠️  No page object available.")
            return
        try:
            current_url = self.page.url if hasattr(self.page, "url") else "(unknown)"
        except Exception as e:
            current_url = f"(error retrieving url: {e})"
        print(f"🔍 Current URL: {current_url}")
        try:
            page_title = await self.page.title()
        except Exception as e:
            page_title = f"(error retrieving title: {e})"
        print(f"🔍 Page title: {page_title}")
        
        # Check for login form elements
        selectors_to_check = [
            '#id_login',
            'input[name="login"]',
            'input[type="text"]',
            'input[placeholder*="username" i]',
            'input[placeholder*="email" i]',
            'form'
        ]
        
        for selector in selectors_to_check:
            count = await self.page.locator(selector).count()
            if count > 0:
                is_visible = await self.page.locator(selector).first.is_visible()
                print(f"🔍 {selector}: exists={count}, visible={is_visible}")
            else:
                print(f"🔍 {selector}: not found")
    
    async def login(self) -> bool:
        """Login to LeetCode with improved reliability"""
        try:
            print("🔐 Attempting to login to LeetCode...")
            start_time = time.time()

            # Step 1: Go to login page directly
            if not self.page:
                print("❌ No page object available for navigation.")
                return False
            try:
                await self.page.goto("https://leetcode.com/accounts/login/", wait_until='networkidle')
            except Exception as e:
                print(f"❌ Failed to navigate to login page: {e}")
                return False
            await asyncio.sleep(3)

            # Debug current state
            await self.debug_page_state()
            
            # Step 2: Find and fill login credentials
            try:
                # Wait for login form to be ready
                await self.page.wait_for_selector('#id_login', timeout=10000, state='visible')
                await self.page.wait_for_selector('#id_password', timeout=10000, state='visible')
                
                print("📝 Filling in credentials...")
                
                # Clear and fill username
                await self.page.fill('#id_login', '')
                await self.page.type('#id_login', self.config.username, delay=100)
                await asyncio.sleep(0.5)
                
                # Clear and fill password
                await self.page.fill('#id_password', '')
                await self.page.type('#id_password', self.config.password, delay=100)
                await asyncio.sleep(0.5)
                
            except Exception as e:
                print(f"❌ Failed to fill credentials: {e}")
                return False
            
            # Step 3: Handle CAPTCHA or other verification if present
            # Add small delay to see if any captcha appears
            await asyncio.sleep(2)
            
            # Step 4: Submit the form using Enter key (most reliable method)
            print("🔄 Submitting login form...")
            try:
                await self.page.focus('#id_password')
                await self.page.keyboard.press('Enter')
                print("✅ Login form submitted")
            except Exception as e:
                print(f"❌ Failed to submit form: {e}")
                return False
            
            # Step 5: Wait for login completion with multiple success indicators
            print("⏳ Waiting for login to complete...")
            
            success_selectors = [
                # Look for user avatar or profile menu
                '[data-cy="navbar-user-dropdown"]',
                'img[alt*="avatar"]',
                '.avatar',
                
                # Look for premium badge
                'text=Premium',
                '[class*="premium"]',
                
                # Look for profile link
                'a[href*="/profile/"]',
                
                # Look for study plan or other authenticated features
                'text=Study Plan',
                
                # Look for problems list or dashboard elements
                '[data-cy="favorite-btn"]',
                '[data-cy="progress-bar"]'
            ]
            
            login_successful = False
            success_indicator = None
            
            # Try to detect successful login
            for selector in success_selectors:
                try:
                    await self.page.wait_for_selector(selector, timeout=15000)
                    login_successful = True
                    success_indicator = selector
                    print(f"✅ Login success indicator found: {selector}")
                    break
                except:
                    continue
            
            # Alternative: check URL change
            if not login_successful:
                await asyncio.sleep(5)  # Wait a bit more
                current_url = self.page.url
                if "/accounts/login/" not in current_url or "?next=" in current_url:
                    login_successful = True
                    success_indicator = "URL change"
                    print("✅ Login successful (URL changed)")
            
            # Final check: look for absence of login form
            if not login_successful:
                try:
                    login_form = await self.page.query_selector('#id_login')
                    if not login_form or not await login_form.is_visible():
                        login_successful = True
                        success_indicator = "Login form disappeared"
                        print("✅ Login successful (login form no longer present)")
                except:
                    pass
            
            if login_successful:
                elapsed_time = time.time() - start_time
                print(f"✅ Login successful ({elapsed_time:.2f}s) - Success indicator: {success_indicator}")
                return True
            else:
                print("❌ Login failed - no success indicators found")
                await self.page.screenshot(path="login_failed_final.png")
                return False
        
        except Exception as e:
            print(f"❌ Login failed with exception: {e}")
            try:
                if self.page is not None:
                    await self.page.screenshot(path="login_exception.png")
            except Exception as screenshot_exc:
                print(f"⚠️ Failed to take screenshot: {screenshot_exc}")
            return False
    async def navigate_to_problem(self, problem_url: str) -> Dict:
        """Navigate to a specific problem and extract details with improved selectors"""
        try:
            print(f"📄 Loading problem: {problem_url}")
            if self.page is None:
                raise Exception("❌ Cannot navigate: self.page is None")
            await self.page.goto(problem_url, wait_until='domcontentloaded')

            # Wait for page to load and ensure main content is present
            # Try to wait for a main content selector, fallback to sleep if not found
            main_content_selectors = [
                '[data-track-load="description_content"]',
                'h1[data-cy="question-title"]',
                '.question-content',
                '[data-cy="question-title"]',
                'h1'
            ]
            found_selector = False
            for selector in main_content_selectors:
                try:
                    await self.page.wait_for_selector(selector, timeout=7000)
                    found_selector = True
                    print(f"✅ Main content selector found: {selector}")
                    break
                except Exception:
                    continue
            if not found_selector:
                print("⚠️ Main content selector not found, waiting extra 3 seconds")
                await asyncio.sleep(3)
            # Updated selectors for problem title based on current LeetCode structure
            title_selectors = [
                # Main title selector for problem page
                'h1[data-cy="question-title"]',
                'h1',
                '[data-cy="question-title"]',
                '.question-title',
                # Alternative selectors
                'div[class*="question-title"] h1',
                'div[class*="title"] h1',
                '.css-v3d350',
                # Fallback selectors
                'span[class*="title"]',
                '[class*="question"] h1'
            ]
            
            title = ""
            for selector in title_selectors:
                try:
                    await self.page.wait_for_selector(selector, timeout=5000)
                    title_element = await self.page.query_selector(selector)
                    if title_element:
                        title = await title_element.text_content()
                        if title and title.strip():
                            title = title.strip()
                            print(f"✅ Found title with selector: {selector}")
                            break
                except:
                    continue
            
            # Updated selectors for problem description
            description_selectors = [
                # Main content selectors
                '[data-track-load="description_content"]',
                '.question-content',
                '[class*="question-content"]',
                '.content__u3I1',
                # Alternative content selectors
                'div[class*="description"]',
                '.question-detail',
                '[class*="problem-statement"]',
                # Fallback selectors
                'div[class*="content"] p',
                '.problem-content',
                '[data-cy="question-content"]'
            ]
            
            description = ""
            for selector in description_selectors:
                try:
                    description_element = await self.page.query_selector(selector)
                    if description_element:
                        desc_text = await description_element.inner_text()
                        if desc_text and len(desc_text.strip()) > 50:  # Ensure we get substantial content
                            description = desc_text.strip()
                            print(f"✅ Found description with selector: {selector}")
                            break
                except:
                    continue
            
            # Updated selectors for difficulty
            difficulty_selectors = [
                # Common difficulty selectors
                '[diff]',
                '.difficulty',
                '[class*="difficulty"]',
                'span[class*="difficulty"]',
                # Text-based difficulty
                'text=Easy',
                'text=Medium', 
                'text=Hard',
                # Alternative selectors
                '[data-difficulty]',
                '.text-difficulty-easy',
                '.text-difficulty-medium',
                '.text-difficulty-hard'
            ]
            
            difficulty = "Unknown"
            for selector in difficulty_selectors:
                try:
                    difficulty_element = await self.page.query_selector(selector)
                    if difficulty_element:
                        diff_text = await difficulty_element.get_attribute("diff")
                        if not diff_text:
                            diff_text = await difficulty_element.text_content()
                        if diff_text and diff_text.strip() in ['Easy', 'Medium', 'Hard']:
                            difficulty = diff_text.strip()
                            print(f"✅ Found difficulty with selector: {selector}")
                            break
                except:
                    continue
            
            # If we still don't have title, try to extract from URL
            if not title or title == "Unknown":
                import re
                url_match = re.search(r'/problems/([^/]+)/', problem_url)
                if url_match:
                    title = url_match.group(1).replace('-', ' ').title()
                    print(f"✅ Extracted title from URL: {title}")
            
            result = {
                "title": title if title else "Unknown Problem",
                "description": description if description else "No description found - please check problem page manually",
                "difficulty": difficulty,  
                "url": problem_url
            }
            
            print(f"✅ Problem loaded: {result['title']} ({result['difficulty']})")
            
            # If we didn't get description, take a screenshot for debugging
            if not description:
                await self.page.screenshot(path="problem_extraction_debug.png")
                print("⚠️ Description not found - screenshot saved for debugging")
            
            return result
            
        except Exception as e:
            print(f"❌ Failed to load problem: {e}")
            if self.page is not None:
                try:
                    await self.page.screenshot(path="problem_load_error.png")
                except Exception as screenshot_error:
                    print(f"⚠️ Failed to take screenshot: {screenshot_error}")
            return {"error": f"Failed to load problem: {e}"}
    
    async def run_tests(self, code: str) -> List[Dict]:
        """Run test cases for the solution with improved reliability"""
        try:
            print("🧪 Running test cases...")
            
            # Wait for page to be fully loaded
            await asyncio.sleep(3)
            
            # Clear and input code with improved editor detection
            await self._clear_and_input_code(code)
            
            # Find and click Run button with updated selectors
            run_selectors = [
                # Updated selectors for current LeetCode interface
                'button[data-e2e-locator="console-run-button"]',
                'button[data-e2e-locator*="run"]',
                'button[data-cy="run-code-btn"]',
                'button[data-cy*="run"]',
                # Text-based selectors
                'button:has-text("Run")',
                'button >> text="Run"',
                # Class-based selectors
                '.run-button',
                'button[class*="run"]',
                # Alternative selectors
                '[data-testid="run-button"]',
                'button[title*="run" i]'
            ]
            
            run_clicked = False
            for selector in run_selectors:
                try:
                    if self.page is None:
                        continue
                    await self.page.wait_for_selector(selector, timeout=5000, state='visible')
                    element = await self.page.query_selector(selector)
                    if element is not None:
                        is_disabled = await element.get_attribute('disabled')
                        if not is_disabled:
                            await element.click()
                            run_clicked = True
                            print(f"✅ Clicked run button: {selector}")
                            break
                except:
                    continue
            
            if not run_clicked:
                print("❌ Could not find or click run button")
                if self.page is not None:
                    try:
                        await self.page.screenshot(path="run_button_error.png")
                    except Exception as screenshot_error:
                        print(f"⚠️ Failed to take screenshot: {screenshot_error}")
                return [{"error": "Run button not found"}]
            
            # Wait for test results with longer timeout
            print("⏳ Waiting for test results...")
            await asyncio.sleep(10)
            
            # Get test results
            return await self._get_test_results()
            
        except Exception as e:
            print(f"❌ Test run failed: {e}")
            return [{"error": f"Test run failed: {e}"}]
    
    async def _clear_and_input_code(self, code: str):
        """Clear editor and input new code with improved reliability"""
        print("📝 Clearing editor and inserting code...")
        
        # Updated editor selectors for current LeetCode
        editor_selectors = [
            # Monaco editor selectors
            '.monaco-editor textarea',
            '.monaco-editor .inputarea',
            '.monaco-editor',
            # CodeMirror selectors
            '.CodeMirror-code',
            '.CodeMirror textarea',
            '.CodeMirror',
            # Generic editor selectors
            '[data-mode-id="python"]',
            '.view-lines',
            '[role="textbox"]',
            # Alternative selectors
            'div[class*="monaco-editor"]',
            'div[class*="editor"]'
        ]
        
        editor_found = False
        if self.page is not None:
            for selector in editor_selectors:
                try:
                    await self.page.wait_for_selector(selector, timeout=3000)
                    await self.page.click(selector)
                    editor_found = True
                    print(f"✅ Found and focused code editor: {selector}")
                    break
                except Exception as e:
                    continue
        else:
            print("❌ self.page is None, cannot focus code editor.")

        if not editor_found:
            print("⚠️ Code editor not found, attempting to continue...")
        # Clear existing code and input new solution
        if self.page is not None and hasattr(self.page, "keyboard") and self.page.keyboard is not None:
            await self.page.keyboard.down("Control")
            await self.page.keyboard.press("A")
            await self.page.keyboard.up("Control")
            await asyncio.sleep(0.5)
            await self.page.keyboard.press("Delete")
            await asyncio.sleep(0.5)
        else:
            print("⚠️ self.page.keyboard is not available, skipping code clearing.")
        # Type the code with delay to avoid issues
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if self.page is not None and hasattr(self.page, "keyboard") and self.page.keyboard is not None:
                for i, line in enumerate(lines):
                    await self.page.keyboard.type(line)
                    if i < len(lines) - 1:  # Don't add newline after last line
                        await self.page.keyboard.press("Enter")
                    await asyncio.sleep(0.1)  # Small delay between lines
                await asyncio.sleep(2)
            else:
                print("⚠️ self.page.keyboard is not available, skipping code typing.")
    async def _get_test_results(self) -> List[Dict]:
        """Extract test results from the page"""
        try:
            # Wait a bit more for results to load
            await asyncio.sleep(5)
            
            # Look for test result indicators
            result_selectors = [
                # Success indicators
                'text=Accepted',
                'text=All test cases passed',
                '.accepted',
                '[class*="accepted"]',
                
                # Failure indicators  
                'text=Wrong Answer',
                'text=Runtime Error',
                'text=Time Limit Exceeded',
                '.error',
                '[class*="error"]',
                
                # Result containers
                '[data-cy="console-result"]',
                '.console-result',
                '[class*="result"]'
            ]
            
            results = []
            for selector in result_selectors:
                try:
                    elements = await self.page.query_selector_all(selector)
                    for element in elements:
                        text = await element.text_content()
                        if text and text.strip():
                            results.append({
                                "status": text.strip(),
                                "selector": selector
                            })
                except:
                    continue
            
            if not results:
                # Fallback: take screenshot and return generic result
                await self.page.screenshot(path="test_results_debug.png")
                results = [{"status": "Test completed - check screenshot for details"}]
            
            return results
            
        except Exception as e:
            return [{"error": f"Failed to get test results: {e}"}]
    
    async def submit_solution(self, code: str) -> Dict:
        """Submit the solution code with improved button detection"""
        try:
            print("🚀 Submitting solution...")
            
            # Clear and input code
            await self._clear_and_input_code(code)
            
            # Find and click submit button with comprehensive selectors
            submit_selectors = [
                # Primary submit button selectors for current LeetCode
                'button[data-e2e-locator="console-submit-button"]',
                'button[data-e2e-locator*="submit"]',
                'button[data-cy="submit-code-btn"]', 
                'button[data-cy*="submit"]',
                
                # Text-based selectors
                'button:has-text("Submit")',
                'button:has-text("Submit Solution")',
                'button >> text="Submit"',
                
                # Class-based selectors
                '.submit-button',
                'button.submit-btn',
                'button[class*="submit"]',
                
                # Alternative selectors
                '[data-testid="submit-button"]',
                'button[title*="submit" i]'
            ]
            
            submit_clicked = False
            for selector in submit_selectors:
                try:
                    if self.page is None:
                        continue
                    await self.page.wait_for_selector(selector, timeout=5000, state='visible')
                    element = await self.page.query_selector(selector)
                    if element is not None:
                        is_disabled = await element.get_attribute('disabled')
                        if not is_disabled:
                            await element.scroll_into_view_if_needed()
                            await asyncio.sleep(0.5)
                            await element.click()
                            submit_clicked = True
                            print(f"✅ Clicked submit button: {selector}")
                            break
                except:
                    continue
            
            if not submit_clicked:
                # Try keyboard shortcut as fallback
                print("🔄 Trying keyboard shortcut...")
                if self.page is not None and hasattr(self.page, "keyboard") and self.page.keyboard is not None:
                    await self.page.keyboard.press("Control+Enter")
                    submit_clicked = True
                    print("✅ Attempted submit via Ctrl+Enter")
                else:
                    print("⚠️ Cannot use keyboard shortcut: self.page or self.page.keyboard is None")
                if self.page is not None:
                    await self.page.screenshot(path="submit_button_error.png")
                return {"error": "Submit button not found"}
            
            # Wait for submission to process
            print("⏳ Waiting for submission to process...")
            await asyncio.sleep(15)  # Increased wait time
            
            # Get submission result
            return await self._get_submission_result()
            
        except Exception as e:
            print(f"❌ Submission failed: {e}")
            if self.page is not None:
                try:
                    await self.page.screenshot(path="submission_exception.png")
                except Exception as screenshot_err:
                    print(f"⚠️ Failed to take screenshot: {screenshot_err}")
            return {"error": f"Submission failed: {e}"}

        async def _get_submission_result(self) -> Dict:
            """Extract submission results from the page"""
        try:
            # Wait for results to appear
            await asyncio.sleep(5)
            
            # Look for submission result indicators
            result_selectors = [
                # Success indicators
                'text=Accepted',
                '.accepted',
                '[class*="accepted"]',
                
                # Failure indicators
                'text=Wrong Answer',
                'text=Runtime Error',
                'text=Time Limit Exceeded',
                'text=Memory Limit Exceeded',
                '.error',
                '[class*="error"]',
                
                # Performance stats
                '[class*="runtime"]',
                '[class*="memory"]',
                '[class*="beats"]'
            ]
            
            result = {"status": "Unknown"}
            
            if self.page is None:
                return {"error": "Page is not initialized"}
            for selector in result_selectors:
                try:
                    elements = await self.page.query_selector_all(selector) if self.page else []
                    for element in elements:
                        if element is None:
                            continue
                        text = await element.text_content()
                        if text and text.strip():
                            if "Accepted" in text:
                                result["status"] = "Accepted"
                            elif any(error in text for error in ["Wrong Answer", "Runtime Error", "Time Limit", "Memory Limit"]):
                                result["status"] = text.strip()
                            elif "ms" in text or "MB" in text or "beats" in text:
                                result["performance"] = text.strip()
                except:
                    continue
            
            # Take screenshot for debugging
            await self.page.screenshot(path="submission_result_debug.png")
            
            return result
            
        except Exception as e:
            return {"error": f"Failed to get submission result: {e}"}

class LeetCodeSolver:
    def __init__(self, api_key: str, model: str = "deepseek-coder", base_url: str = "https://api.deepseek.com"):
        """
        Initialize with DeepSeek API configuration
        
        Args:
            api_key: DeepSeek API key
            model: DeepSeek model name (e.g., "deepseek-coder", "deepseek-chat")
            base_url: DeepSeek API base URL
        """
        
        self.llm = ChatOpenAI(
        api_key=api_key,
        model=model,
        temperature=0.5,
        base_url=base_url
   )
    
    async def solve_problem(self, problem_description: str, problem_title: str) -> str:
        """Generate solution code for the problem"""
        prompt = f"""
        You are an expert competitive programmer. Solve this LeetCode problem:
        
        Title: {problem_title}
        Description: {problem_description}
        
        Requirements:
        1. Provide a complete, working Python solution
        2. Include proper class and method structure as expected by LeetCode
        3. Optimize for both time and space complexity
        4. Add brief comments explaining the approach
        5. Only return the code, no additional explanation
        
        Code:
        """
        
        print("Calling DeepSeek API...")
        try:
            response = await asyncio.wait_for(
                self.llm.ainvoke([HumanMessage(content=prompt)]),
                timeout=60  # seconds
            )
            print("DeepSeek API call completed.")
            # Ensure we always return a string, even if response.content is a list or dict
            if isinstance(response.content, str):
                return response.content
            else:
                # Convert list/dict to string (e.g., JSON)
                import json
                return json.dumps(response.content)
        except Exception as e:
            print(f"DeepSeek API call failed: {e}")
            return "# DeepSeek API call failed"
    async def optimize_solution(self, original_code: str, runtime_stats: Dict, problem_description: str) -> str:
        """Suggest optimizations based on runtime performance"""
        prompt = f"""
        Analyze this LeetCode solution and suggest optimizations:
        
        Original Code:
        {original_code}
        
        Runtime Stats:
        {runtime_stats}
        
        Problem Description:
        {problem_description}
        Please provide:
        1. An optimized version of the code if possible
        2. Explanation of optimizations made
        3. Expected time/space complexity improvements
        
        Focus on improving runtime performance while maintaining correctness.
        """
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            # Ensure we always return a string, even if response.content is a list or dict
            if isinstance(response.content, str):
                return response.content
            else:
                import json
                return json.dumps(response.content)
        except Exception as e:
            print(f"LLM API call failed: {e}")
            return "# LLM API call failed"

# Jupyter-compatible wrapper class
class JupyterLeetCodeSolver:
    def __init__(self, leetcode_config: LeetCodeConfig, solver: LeetCodeSolver):
        self.leetcode_config = leetcode_config
        self.solver = solver
    
    def run_async(self, coro):
        """Helper method to run async functions in Jupyter"""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro)
    
    def solve_problem(self, problem_url: str) -> Dict:
        """Jupyter-compatible method to solve a LeetCode problem"""
        return self.run_async(self._solve_problem_async(problem_url))
    
    async def _solve_problem_async(self, problem_url: str) -> Dict:
        """Internal async method"""
        try:
            # Step 1: Fetch problem details
            print("🔍 Fetching problem details...")
            problem_details = await self._fetch_problem(problem_url)
            
            if "error" in problem_details:
                return {"error": problem_details["error"]}
            
            print(f"📋 Problem: {problem_details['title']} ({problem_details['difficulty']})")
            
            # Step 2: Generate solution
            print("🤖 Generating solution...")
            solution_code = await self.solver.solve_problem(
                problem_details["description"],
                problem_details["title"]
            )
            
            print("✅ Solution generated!")
            print(f"Generated code:\n{solution_code}")
            
            # Step 3: Run tests
            print("🧪 Running tests...")
            test_results = await self._run_tests(problem_url, solution_code)
            
            # Step 4: Submit solution
            print("🚀 Submitting solution...")
            submission_result = await self._submit_solution(problem_url, solution_code)
            
            result = {
                "problem_details": problem_details,
                "solution_code": solution_code,
                "test_results": test_results,
                "submission_result": submission_result,
                "status": "Completed"
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Solver failed: {e}"}
    
    async def _fetch_problem(self, problem_url: str) -> Dict:
        """Fetch problem details"""
        async with LeetCodeAutomation(self.leetcode_config) as automation:
            if not await automation.login():
                return {"error": "Login failed"}
            return await automation.navigate_to_problem(problem_url)
    
    async def _run_tests(self, problem_url: str, solution_code: str) -> List[Dict]:
        """Run test cases"""
        async with LeetCodeAutomation(self.leetcode_config) as automation:
            if not await automation.login():
                return [{"error": "Login failed"}]
            await automation.navigate_to_problem(problem_url)
            return await automation.run_tests(solution_code)
    
    async def _submit_solution(self, problem_url: str, solution_code: str) -> Dict:
        """Submit solution"""
        async with LeetCodeAutomation(self.leetcode_config) as automation:
            if not await automation.login():
                return {"error": "Login failed"}
            await automation.navigate_to_problem(problem_url)
            return await automation.submit_solution(solution_code)

# Quick setup function with DeepSeek API support
def quick_setup(username: str, password: str, deepseek_api_key: str, 
               model: str = "deepseek-coder", headless: bool = False):
    """
    Quick setup function for LeetCode solver with DeepSeek API
    
    Args:
        username: LeetCode username
        password: LeetCode password  
        deepseek_api_key: DeepSeek API key
        model: DeepSeek model to use (default: "deepseek-coder")
        headless: Whether to run browser in headless mode (default: False for debugging)
    
    Returns:
        JupyterLeetCodeSolver instance ready to use
    """
    config = LeetCodeConfig(
        username=username,
        password=password,
        headless=headless,
        timeout=60000,
        max_attempts=3
    )
    
    # Initialize with DeepSeek configuration
    solver = LeetCodeSolver(
        api_key=deepseek_api_key, 
        model=model,
        base_url="https://api.deepseek.com"
    )
    
    return JupyterLeetCodeSolver(config, solver)

print("🚀 LeetCode Solver with DeepSeek API ready! Use quick_setup() to get started.")


