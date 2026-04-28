#!/usr/bin/env python3
"""
Pre-Submission Verification Script
Validates all hackathon submission requirements
"""

import os
import sys
import json
import yaml
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Color output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str) -> None:
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}\n")

def print_check(name: str, status: bool, details: str = "") -> None:
    symbol = f"{Colors.GREEN}✓{Colors.RESET}" if status else f"{Colors.RED}✗{Colors.RESET}"
    msg = f"  {symbol} {name}"
    if details:
        msg += f" ({details})"
    print(msg)

def check_file_exists(path: str, name: str) -> Tuple[bool, str]:
    """Check if a file exists"""
    if os.path.exists(path):
        size = os.path.getsize(path)
        return True, f"{size} bytes"
    return False, "MISSING"

def check_inference_py() -> Dict[str, Any]:
    """Verify inference.py compliance"""
    print_header("1. INFERENCE.PY VERIFICATION")
    results = {}
    
    # Check file exists
    exists, detail = check_file_exists("inference.py", "inference.py")
    print_check("inference.py exists in root", exists, detail)
    results["file_exists"] = exists
    
    if not exists:
        return results
    
    # Read file
    with open("inference.py", "r") as f:
        content = f.read()
    
    # Check for required env vars
    has_api_base = "API_BASE_URL" in content
    has_model_name = "MODEL_NAME" in content
    has_openai_key = "OPENAI_API_KEY" in content
    
    print_check("Reads API_BASE_URL from environment", has_api_base)
    print_check("Reads MODEL_NAME from environment", has_model_name)
    print_check("Reads OPENAI_API_KEY from environment", has_openai_key)
    
    results["env_vars"] = has_api_base and has_model_name and has_openai_key
    
    # Check for [START], [STEP], [END] format
    has_start = "[START]" in content
    has_step = "[STEP]" in content
    has_end = "[END]" in content
    
    print_check("Emits [START] format token", has_start)
    print_check("Emits [STEP] format token", has_step)
    print_check("Emits [END] format token", has_end)
    
    results["output_format"] = has_start and has_step and has_end
    
    # Check for OpenAI client usage
    has_openai = "from openai" in content or "OpenAI(" in content
    print_check("Uses OpenAI Client", has_openai)
    results["uses_openai"] = has_openai
    
    return results

def check_openenv_yaml() -> Dict[str, Any]:
    """Verify OpenEnv spec compliance"""
    print_header("2. OPENENV.YAML COMPLIANCE")
    results = {}
    
    exists, detail = check_file_exists("openenv.yaml", "openenv.yaml")
    print_check("openenv.yaml exists", exists, detail)
    
    if not exists:
        return results
    
    try:
        with open("openenv.yaml", "r") as f:
            spec = yaml.safe_load(f)
        
        # Check required fields
        has_name = "name" in spec
        has_version = "version" in spec
        has_pool_size = "pool_size" in spec
        has_tasks = "tasks" in spec
        
        print_check("Has 'name' field", has_name, spec.get("name", "N/A"))
        print_check("Has 'version' field", has_version, spec.get("version", "N/A"))
        print_check("Has 'pool_size' field", has_pool_size, f"pool_size={spec.get('pool_size', 'N/A')}")
        print_check("Has 'tasks' field", has_tasks)
        
        results["structure"] = all([has_name, has_version, has_pool_size, has_tasks])
        
        # Check tasks
        if has_tasks:
            tasks = spec.get("tasks", [])
            task_ids = [t.get("id") for t in tasks]
            
            has_easy = "easy" in task_ids
            has_medium = "medium" in task_ids
            has_hard = "hard" in task_ids
            
            print_check("Has 'easy' task", has_easy)
            print_check("Has 'medium' task", has_medium)
            print_check("Has 'hard' task", has_hard)
            print(f"      Found {len(task_ids)} total tasks: {task_ids}")
            
            results["tasks"] = all([has_easy, has_medium, has_hard])
        
        # Check pool size
        pool_size = spec.get("pool_size")
        print_check("Pool size defined", pool_size is not None, f"size={pool_size}")
        results["pool_size"] = pool_size is not None
        
        results["valid"] = True
    except Exception as e:
        print_check("YAML parsing", False, str(e))
        results["valid"] = False
    
    return results

def check_typed_models() -> Dict[str, Any]:
    """Verify Pydantic models are typed"""
    print_header("3. TYPED MODELS VERIFICATION")
    results = {}
    
    model_path = "environment/models.py"
    exists, detail = check_file_exists(model_path, "models.py")
    print_check("environment/models.py exists", exists, detail)
    
    if not exists:
        return results
    
    try:
        with open(model_path, "r") as f:
            content = f.read()
        
        # Check for Pydantic v2
        has_pydantic_v2 = "from pydantic import" in content or "BaseModel" in content
        print_check("Uses Pydantic models", has_pydantic_v2)
        
        # Check for type annotations
        has_type_hints = ": " in content and "str" in content
        print_check("Uses type annotations", has_type_hints)
        
        # Count classes
        class_count = content.count("class ")
        print_check("Has typed model classes", class_count > 0, f"{class_count} classes")
        
        results["valid"] = has_pydantic_v2 and has_type_hints
    except Exception as e:
        print_check("Model validation", False, str(e))
        results["valid"] = False
    
    return results

def check_endpoints() -> Dict[str, Any]:
    """Verify OpenEnv endpoints"""
    print_header("4. OPENENV ENDPOINTS VERIFICATION")
    results = {}
    
    app_path = "app.py"
    exists, detail = check_file_exists(app_path, "app.py")
    print_check("app.py exists", exists, detail)
    
    if not exists:
        return results
    
    try:
        with open(app_path, "r") as f:
            content = f.read()
        
        has_reset = "def reset" in content
        has_step = "def step" in content
        has_state = "def state" in content
        has_health = "def health" in content or "/health" in content
        
        print_check("Has /reset endpoint", has_reset)
        print_check("Has /step endpoint", has_step)
        print_check("Has /state endpoint", has_state)
        print_check("Has /health endpoint", has_health)
        
        results["valid"] = all([has_reset, has_step, has_state, has_health])
    except Exception as e:
        print_check("Endpoint validation", False, str(e))
        results["valid"] = False
    
    return results

def check_dockerfile() -> Dict[str, Any]:
    """Verify Dockerfile"""
    print_header("5. DOCKERFILE VERIFICATION")
    results = {}
    
    exists, detail = check_file_exists("Dockerfile", "Dockerfile")
    print_check("Dockerfile exists", exists, detail)
    
    if not exists:
        return results
    
    try:
        with open("Dockerfile", "r") as f:
            content = f.read()
        
        has_python = "python" in content.lower()
        has_port = "7860" in content
        has_expose = "EXPOSE" in content
        has_cmd = "CMD" in content
        
        print_check("Uses Python base image", has_python)
        print_check("Exposes port 7860", has_port)
        print_check("Has EXPOSE instruction", has_expose)
        print_check("Has CMD instruction", has_cmd)
        
        results["valid"] = all([has_python, has_port, has_expose, has_cmd])
    except Exception as e:
        print_check("Dockerfile validation", False, str(e))
        results["valid"] = False
    
    return results

def check_requirements() -> Dict[str, Any]:
    """Verify requirements.txt"""
    print_header("6. REQUIREMENTS.TXT VERIFICATION")
    results = {}
    
    exists, detail = check_file_exists("requirements.txt", "requirements.txt")
    print_check("requirements.txt exists", exists, detail)
    
    if not exists:
        return results
    
    try:
        with open("requirements.txt", "r") as f:
            content = f.read()
        
        has_fastapi = "fastapi" in content.lower()
        has_pydantic = "pydantic" in content.lower()
        has_openai = "openai" in content.lower()
        has_pytest = "pytest" in content.lower()
        
        print_check("Has fastapi", has_fastapi)
        print_check("Has pydantic", has_pydantic)
        print_check("Has openai", has_openai)
        print_check("Has pytest", has_pytest)
        
        # Count lines
        line_count = len([l for l in content.split('\n') if l.strip() and not l.startswith('#')])
        print_check("Dependencies defined", line_count > 0, f"{line_count} packages")
        
        results["valid"] = all([has_fastapi, has_pydantic, has_openai, has_pytest])
    except Exception as e:
        print_check("Requirements validation", False, str(e))
        results["valid"] = False
    
    return results

def check_tasks_and_graders() -> Dict[str, Any]:
    """Verify 3+ tasks and graders"""
    print_header("7. TASKS & GRADERS VERIFICATION")
    results = {}
    
    # Check grader exists
    grader_path = "environment/grader.py"
    exists, detail = check_file_exists(grader_path, "grader.py")
    print_check("grader.py exists", exists, detail)
    
    if not exists:
        return results
    
    try:
        with open(grader_path, "r") as f:
            content = f.read()
        
        # Check for grader functions
        has_grade_func = "def grade" in content or "def score" in content
        print_check("Has grading function", has_grade_func)
        
        # Verify openenv.yaml tasks
        with open("openenv.yaml", "r") as f:
            spec = yaml.safe_load(f)
        
        tasks = spec.get("tasks", [])
        print_check("Has 3+ tasks", len(tasks) >= 3, f"{len(tasks)} tasks")
        
        for task in tasks:
            task_id = task.get("id", "unknown")
            max_steps = task.get("max_steps")
            score_range = task.get("expected_score_range", [])
            print(f"      - Task '{task_id}': {max_steps} steps, score range {score_range}")
        
        results["valid"] = has_grade_func and len(tasks) >= 3
    except Exception as e:
        print_check("Tasks validation", False, str(e))
        results["valid"] = False
    
    return results

def check_test_suite() -> Dict[str, Any]:
    """Verify test suite exists"""
    print_header("8. TEST SUITE VERIFICATION")
    results = {}
    
    test_dir = "tests"
    if os.path.isdir(test_dir):
        test_files = [f for f in os.listdir(test_dir) if f.startswith("test_") and f.endswith(".py")]
        print_check("Test directory exists", True)
        print_check("Test files found", len(test_files) > 0, f"{len(test_files)} test files")
        
        for test_file in test_files[:5]:
            print(f"      - {test_file}")
        if len(test_files) > 5:
            print(f"      - ... and {len(test_files)-5} more")
        
        results["valid"] = len(test_files) > 0
    else:
        print_check("Test directory exists", False)
        results["valid"] = False
    
    return results

def check_git_status() -> Dict[str, Any]:
    """Check git commit history"""
    print_header("9. GIT REPOSITORY STATUS")
    results = {}
    
    try:
        # Check if .git exists
        if os.path.isdir(".git"):
            print_check(".git directory exists", True)
            
            # Get latest commit
            result = subprocess.run(
                ["git", "log", "--oneline", "-1"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                commit_msg = result.stdout.strip()
                print(f"      Latest commit: {commit_msg}")
                print_check("Git history present", True)
                results["valid"] = True
            else:
                print_check("Git history present", False)
                results["valid"] = False
        else:
            print_check(".git directory exists", False)
            results["valid"] = False
    except Exception as e:
        print_check("Git status check", False, str(e))
        results["valid"] = False
    
    return results

def check_environment_variables() -> Dict[str, Any]:
    """Check if required env vars are set"""
    print_header("10. ENVIRONMENT VARIABLES VERIFICATION")
    results = {}
    
    api_base = os.environ.get("API_BASE_URL", "").strip()
    model_name = os.environ.get("MODEL_NAME", "").strip()
    openai_api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    
    has_api_base = len(api_base) > 0
    has_model = len(model_name) > 0
    has_token = len(openai_api_key) > 0
    
    print_check("API_BASE_URL set", has_api_base, "✓" if has_api_base else "MISSING")
    print_check("MODEL_NAME set", has_model, "✓" if has_model else "MISSING")
    print_check("OPENAI_API_KEY set", has_token, "✓" if has_token else "MISSING")
    
    results["valid"] = has_api_base and has_model
    results["token_set"] = has_token
    
    return results

def generate_summary(all_results: Dict[str, Dict]) -> None:
    """Generate final summary"""
    print_header("FINAL SUMMARY")
    
    checklist = [
        ("Inference.py exists with env vars & format", all_results.get("inference", {}).get("file_exists", False)),
        ("OpenEnv.yaml valid with 3+ tasks", all_results.get("openenv", {}).get("valid", False)),
        ("Pydantic models properly typed", all_results.get("models", {}).get("valid", False)),
        ("Required endpoints present", all_results.get("endpoints", {}).get("valid", False)),
        ("Dockerfile valid", all_results.get("dockerfile", {}).get("valid", False)),
        ("Requirements.txt complete", all_results.get("requirements", {}).get("valid", False)),
        ("3+ tasks with graders", all_results.get("tasks", {}).get("valid", False)),
        ("Test suite present", all_results.get("tests", {}).get("valid", False)),
        ("Git repository initialized", all_results.get("git", {}).get("valid", False)),
    ]
    
    passed = sum(1 for _, status in checklist if status)
    total = len(checklist)
    
    for name, status in checklist:
        symbol = f"{Colors.GREEN}✓{Colors.RESET}" if status else f"{Colors.RED}✗{Colors.RESET}"
        print(f"  {symbol} {name}")
    
    print(f"\n  {Colors.BOLD}Score: {passed}/{total} checks passed{Colors.RESET}")
    
    if passed == total:
        print(f"\n  {Colors.GREEN}{Colors.BOLD}🎉 SUBMISSION READY!{Colors.RESET}")
        print(f"  {Colors.GREEN}All pre-submission checks passed.{Colors.RESET}")
    else:
        print(f"\n  {Colors.YELLOW}{Colors.BOLD}⚠️  {total-passed} issue(s) to fix{Colors.RESET}")
    
    print(f"\n  {Colors.YELLOW}Environment Variables Check:{Colors.RESET}")
    env_valid = all_results.get("env_vars", {}).get("valid", False)
    token_set = all_results.get("env_vars", {}).get("token_set", False)
    
    if not env_valid:
        print(f"    {Colors.RED}✗ Required: API_BASE_URL and MODEL_NAME{Colors.RESET}")
    else:
        print(f"    {Colors.GREEN}✓ API_BASE_URL and MODEL_NAME set{Colors.RESET}")
    
    if not token_set:
        print(f"    {Colors.YELLOW}⚠ OPENAI_API_KEY not set{Colors.RESET}")
    else:
        print(f"    {Colors.GREEN}✓ OPENAI_API_KEY set{Colors.RESET}")

def main() -> None:
    """Run all verification checks"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("╔════════════════════════════════════════════════════════════════════════════════╗")
    print("║          MLAuditBench Pre-Submission Verification Checklist                   ║")
    print("║                      Hackathon Requirements Validator                         ║")
    print("╚════════════════════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.RESET}")
    
    # Change to submission directory if running from parent
    if os.path.exists("submission") and os.path.isdir("submission"):
        os.chdir("submission")
    
    all_results = {
        "inference": check_inference_py(),
        "openenv": check_openenv_yaml(),
        "models": check_typed_models(),
        "endpoints": check_endpoints(),
        "dockerfile": check_dockerfile(),
        "requirements": check_requirements(),
        "tasks": check_tasks_and_graders(),
        "tests": check_test_suite(),
        "git": check_git_status(),
        "env_vars": check_environment_variables(),
    }
    
    generate_summary(all_results)
    
    print(f"\n{Colors.BOLD}Required Environment Setup:{Colors.RESET}")
    print(f"  export API_BASE_URL='https://api.openai.com/v1'")
    print(f"  export MODEL_NAME='gpt-4.1-mini'")
    print(f"  export OPENAI_API_KEY='your-openai-api-key'")
    print()

if __name__ == "__main__":
    main()
