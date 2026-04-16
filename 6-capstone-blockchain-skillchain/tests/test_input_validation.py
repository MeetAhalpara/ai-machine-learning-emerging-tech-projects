"""
test_input_validation.py
Automated unit tests tracking repository pathway validation behavior.
"""
import re

def validate_repository_source(input_string):
    input_string = input_string.strip()
    github_pattern = r"^https?://(www\.)?github\.com/[\w-]+/[\w.-]+/?$"
    if re.match(github_pattern, input_string):
        return "REMOTE_GITHUB"
    if input_string.startswith("/") or (len(input_string) > 1 and input_string[1] == ":"):
        return "LOCAL_FILESYSTEM"
    return "INVALID_FORMAT"

def test_valid_github_url_parsing():
    assert validate_repository_source("https://github.com/MeetAhalpara/ai-machine-learning-emerging-tech-projects") == "REMOTE_GITHUB"

def test_valid_local_path_parsing():
    assert validate_repository_source("C:\\Users\\Meeta\\Documents\\B-portfolio-clean-repo") == "LOCAL_FILESYSTEM"

def test_malformed_input_handling():
    assert validate_repository_source("not-a-valid-path-or-url") == "INVALID_FORMAT"